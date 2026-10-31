"""FLUX.2 Klein LoRA trainer."""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor, nn

from src.trainer.base_trainer import TrainContext
from src.trainer.config import FluxFlowConfig, FluxQuantization
from src.trainer.flux.base import FluxBaseTrainer
from src.trainer.flux.flow import (
    add_flow_noise,
    flux2_image_ids,
    flux2_pack,
    flux2_patchify,
    flux2_text_ids,
    flux2_unpack,
    sample_sigmas,
)
from src.trainer.flux.loading import (
    load_flux2_transformer,
    load_flux2_vae,
    load_qwen3,
    load_tokenizer,
)
from src.trainer.flux.lora import FLUX2_TRANSFORMER_TARGETS, QWEN3_TARGETS
from src.trainer.flux.model_files import (
    resolve_text_encoder_files,
    resolve_weights_file,
)
from src.trainer.sdxl.model_loader import resolve_vae_dtype

QWEN3_HIDDEN_STATES_LAYERS = (9, 18, 27)


def format_qwen3_prompt(tokenizer, caption: str) -> str:
    """Chat-template a caption the way Klein expects (thinking disabled)."""
    messages = [{"role": "user", "content": caption}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return (
        f"<|im_start|>user\n{caption}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


class Flux2KleinLoRATrainer(FluxBaseTrainer):
    _tag = "flux2"
    _model_label = "FLUX.2 Klein"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self._config.flux2_klein is not None, "flux2_klein config section is required"
        self._klein = self._config.flux2_klein
        self._tokenizer = None

    @property
    def _quantization(self) -> FluxQuantization:
        return self._klein.quantization

    @property
    def _flow(self) -> FluxFlowConfig:
        return self._klein.flow

    @property
    def _transformer_targets(self) -> str:
        return FLUX2_TRANSFORMER_TARGETS

    @property
    def _text_encoder_targets(self) -> dict[str, list[str]]:
        return {"text_encoder_1": QWEN3_TARGETS, "text_encoder_2": []}

    @property
    def _text_cache_tag(self) -> str:
        return f"flux2_qwen3_{self._klein.text_sequence_length}"

    def _load_components(
        self, ctx: TrainContext
    ) -> tuple[nn.Module, nn.Module, nn.Module, Optional[nn.Module]]:
        cfg = self._klein
        compute_dtype = torch.bfloat16
        qwen_weights, qwen_dir = resolve_text_encoder_files("Qwen3", cfg.qwen3)
        self._tokenizer = load_tokenizer(qwen_dir)
        qwen = load_qwen3(
            qwen_weights, qwen_dir, compute_dtype=compute_dtype, quantization=cfg.quantization
        )
        vae = load_flux2_vae(
            resolve_weights_file("FLUX.2 VAE", cfg.vae_path),
            dtype=resolve_vae_dtype(ctx.vae_dtype),
        )
        transformer = load_flux2_transformer(
            resolve_weights_file("FLUX.2 Klein transformer", cfg.transformer_path),
            quantization=cfg.quantization,
            compute_dtype=compute_dtype,
            device=ctx.device,
        )
        return transformer, vae, qwen, None

    # ----------------------------------------------------------------- text encoding

    def _encode_text(self, captions: list[str], ctx: TrainContext) -> tuple[Tensor]:
        """Returns ``(hidden [B, L, 3 * hidden_size],)`` from Qwen3 layers 9/18/27."""
        config = ctx.config
        device = ctx.device
        assert self._text_encoder_1 is not None
        texts = [format_qwen3_prompt(self._tokenizer, c) for c in captions]
        tokens = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self._klein.text_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        grad_ctx = contextlib.nullcontext() if config.text_encoder_1.train else torch.no_grad()
        with grad_ctx, torch.autocast(device_type=device.type, dtype=ctx.weight_dtype):
            out = self._text_encoder_1(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = torch.cat([out.hidden_states[k] for k in QWEN3_HIDDEN_STATES_LAYERS], dim=2)
        return (hidden.to(dtype=ctx.weight_dtype),)

    # --------------------------------------------------------------------- the step

    @staticmethod
    def _scale_latents(vae: nn.Module, patchified: Tensor) -> Tensor:
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(patchified.device, patchified.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            patchified.device, patchified.dtype
        )
        return (patchified - mean) / std

    def _flow_loss(self, latents: Tensor, text: tuple[Tensor, ...], ctx: TrainContext) -> Tensor:
        assert self._vae is not None and self._forward_transformer is not None
        device = ctx.device
        dtype = ctx.weight_dtype
        (text_hidden,) = text
        flow = self._flow

        x = self._scale_latents(self._vae, flux2_patchify(latents.to(device, torch.float32))).to(dtype)
        batch_size, _, height, width = x.shape

        with torch.no_grad():
            noise = torch.randn_like(x)
            sigmas = sample_sigmas(batch_size, flow.timestep_distribution, flow.timestep_shift, device)
            noisy = add_flow_noise(x, noise, sigmas)
        target = noise - x

        guidance = None
        if self._guidance_embeds:
            guidance = torch.full((batch_size,), flow.guidance_scale, device=device, dtype=dtype)

        with torch.autocast(device_type=device.type, dtype=dtype):
            packed_pred = self._forward_transformer(
                hidden_states=flux2_pack(noisy),
                encoder_hidden_states=text_hidden,
                timestep=sigmas.to(dtype),
                img_ids=flux2_image_ids(batch_size, height, width, device),
                txt_ids=flux2_text_ids(batch_size, text_hidden.shape[1], device),
                guidance=guidance,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        pred = flux2_unpack(packed_pred, height, width)
        return torch.nn.functional.mse_loss(pred.float(), target.float())
