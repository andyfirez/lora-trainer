"""Flux.1 (dev / schnell) LoRA trainer."""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor, nn

from src.trainer.base_trainer import TrainContext
from src.trainer.config import Flux1Variant, FluxFlowConfig, FluxQuantization
from src.trainer.flux.base import FluxBaseTrainer
from src.trainer.flux.flow import (
    add_flow_noise,
    flux1_image_ids,
    flux1_pack,
    flux1_unpack,
    sample_sigmas,
)
from src.trainer.flux.loading import (
    load_clip_l,
    load_flux1_transformer,
    load_flux1_vae,
    load_t5,
    load_tokenizer,
)
from src.trainer.flux.lora import CLIP_TARGETS, FLUX1_TRANSFORMER_TARGETS, T5_TARGETS
from src.trainer.flux.model_files import (
    resolve_text_encoder_files,
    resolve_weights_file,
)
from src.trainer.sdxl.model_loader import resolve_vae_dtype

CLIP_MAX_LENGTH = 77


class Flux1LoRATrainer(FluxBaseTrainer):
    _tag = "flux1"
    _model_label = "Flux.1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self._config.flux1 is not None, "flux1 config section is required"
        self._flux1 = self._config.flux1
        self._clip_tokenizer = None
        self._t5_tokenizer = None

    @property
    def _quantization(self) -> FluxQuantization:
        return self._flux1.quantization

    @property
    def _flow(self) -> FluxFlowConfig:
        return self._flux1.flow

    @property
    def _transformer_targets(self) -> str:
        return FLUX1_TRANSFORMER_TARGETS

    @property
    def _text_encoder_targets(self) -> dict[str, list[str]]:
        return {"text_encoder_1": CLIP_TARGETS, "text_encoder_2": T5_TARGETS}

    @property
    def _text_cache_tag(self) -> str:
        return f"flux1_t5{self._flux1.t5_sequence_length}"

    def _load_components(
        self, ctx: TrainContext
    ) -> tuple[nn.Module, nn.Module, nn.Module, Optional[nn.Module]]:
        cfg = self._flux1
        compute_dtype = torch.bfloat16
        clip_weights, clip_dir = resolve_text_encoder_files("CLIP-L", cfg.clip_l)
        t5_weights, t5_dir = resolve_text_encoder_files("T5-XXL", cfg.t5)
        self._clip_tokenizer = load_tokenizer(clip_dir)
        self._t5_tokenizer = load_tokenizer(t5_dir)

        clip = load_clip_l(clip_weights, clip_dir, compute_dtype=compute_dtype)
        t5 = load_t5(t5_weights, t5_dir, compute_dtype=compute_dtype, quantization=cfg.quantization)
        vae = load_flux1_vae(
            resolve_weights_file("Flux.1 VAE", cfg.vae_path),
            dtype=resolve_vae_dtype(ctx.vae_dtype),
        )
        transformer = load_flux1_transformer(
            resolve_weights_file("Flux.1 transformer", cfg.transformer_path),
            quantization=cfg.quantization,
            compute_dtype=compute_dtype,
            device=ctx.device,
        )
        if cfg.variant == Flux1Variant.DEV and not getattr(transformer.config, "guidance_embeds", False):
            ctx.log.warning("variant=dev but the transformer has no guidance embedder (looks like schnell)")
        if cfg.variant == Flux1Variant.SCHNELL and getattr(transformer.config, "guidance_embeds", False):
            ctx.log.warning("variant=schnell but the transformer has a guidance embedder (looks like dev)")
        return transformer, vae, clip, t5

    # ----------------------------------------------------------------- text encoding

    def _encode_text(self, captions: list[str], ctx: TrainContext) -> tuple[Tensor, Tensor]:
        """Returns ``(t5_hidden [B, L, 4096], clip_pooled [B, 768])``."""
        config = ctx.config
        device = ctx.device
        assert self._text_encoder_1 is not None and self._text_encoder_2 is not None
        clip_tokens = self._clip_tokenizer(
            captions,
            padding="max_length",
            max_length=CLIP_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        t5_tokens = self._t5_tokenizer(
            captions,
            padding="max_length",
            max_length=self._flux1.t5_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_ctx = contextlib.nullcontext() if config.text_encoder_1.train else torch.no_grad()
        t5_ctx = contextlib.nullcontext() if config.text_encoder_2.train else torch.no_grad()
        with clip_ctx, torch.autocast(device_type=device.type, dtype=ctx.weight_dtype):
            clip_out = self._text_encoder_1(clip_tokens.input_ids.to(device), output_hidden_states=False)
            pooled = clip_out.pooler_output.to(dtype=ctx.weight_dtype)
        with t5_ctx, torch.autocast(device_type=device.type, dtype=ctx.weight_dtype):
            t5_out = self._text_encoder_2(t5_tokens.input_ids.to(device))[0].to(dtype=ctx.weight_dtype)
        return t5_out, pooled

    # --------------------------------------------------------------------- the step

    def _flow_loss(self, latents: Tensor, text: tuple[Tensor, ...], ctx: TrainContext) -> Tensor:
        assert self._vae is not None and self._forward_transformer is not None
        device = ctx.device
        dtype = ctx.weight_dtype
        t5_hidden, pooled = text
        flow = self._flow

        vae_cfg = self._vae.config
        x = ((latents.to(device) - vae_cfg.shift_factor) * vae_cfg.scaling_factor).to(dtype)
        batch_size, _, height, width = x.shape

        with torch.no_grad():
            noise = torch.randn_like(x)
            sigmas = sample_sigmas(batch_size, flow.timestep_distribution, flow.timestep_shift, device)
            noisy = add_flow_noise(x, noise, sigmas)
        target = noise - x

        guidance = None
        if self._guidance_embeds:
            guidance = torch.full((batch_size,), flow.guidance_scale, device=device, dtype=dtype)

        img_ids = flux1_image_ids(height, width, device, dtype)
        txt_ids = torch.zeros(t5_hidden.shape[1], 3, device=device, dtype=dtype)

        with torch.autocast(device_type=device.type, dtype=dtype):
            packed_pred = self._forward_transformer(
                hidden_states=flux1_pack(noisy),
                timestep=sigmas.to(dtype),
                guidance=guidance,
                pooled_projections=pooled,
                encoder_hidden_states=t5_hidden,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        pred = flux1_unpack(packed_pred, height, width)
        return torch.nn.functional.mse_loss(pred.float(), target.float())
