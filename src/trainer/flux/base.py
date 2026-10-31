"""Shared trainer logic for Flux.1 and FLUX.2 Klein (flow-matching DiT, cached embeddings)."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
from peft import get_peft_model
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

from src.trainer.base_trainer import BaseLoRATrainer, CacheProgress, TrainContext
from src.trainer.config import FluxFlowConfig, FluxQuantization, TrainConfig
from src.trainer.flux.cache import build_flux_latent_cache, build_flux_text_cache
from src.trainer.flux.lora import (
    apply_flux_resume_state,
    collect_flux_resume_state,
    export_flux_lora_state_dict,
    write_flux_lora,
)
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config
from src.trainer.sdxl.mixed_precision import cast_trainable_params_to_fp32

_PIXEL_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


class FluxBaseTrainer(BaseLoRATrainer):
    """Template for Flux-family trainers.

    Part mapping: ``unet`` config = transformer, ``text_encoder_1`` / ``text_encoder_2`` =
    first / second text encoder (CLIP-L + T5 for Flux.1, Qwen3 + none for Klein).
    """

    _tag: str = "flux"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transformer: Optional[nn.Module] = None  # PEFT-wrapped (exported / resumed)
        self._forward_transformer: Optional[nn.Module] = None  # possibly torch.compile'd
        self._vae: Optional[nn.Module] = None
        self._text_encoder_1: Optional[nn.Module] = None
        self._text_encoder_2: Optional[nn.Module] = None
        self._guidance_embeds = False
        self._latent_cache: Optional[dict[str, Tensor]] = None
        self._text_cache: Optional[dict[str, tuple[Tensor, ...]]] = None

    # ------------------------------------------------------------ subclass contract

    @property
    @abstractmethod
    def _quantization(self) -> FluxQuantization: ...

    @property
    @abstractmethod
    def _flow(self) -> FluxFlowConfig: ...

    @property
    @abstractmethod
    def _transformer_targets(self) -> str | list[str]: ...

    @property
    @abstractmethod
    def _text_encoder_targets(self) -> dict[str, list[str]]:
        """Part name (``text_encoder_1`` / ``text_encoder_2``) -> LoRA target module names."""

    @property
    @abstractmethod
    def _text_cache_tag(self) -> str: ...

    @abstractmethod
    def _load_components(
        self, ctx: TrainContext
    ) -> tuple[nn.Module, nn.Module, nn.Module, Optional[nn.Module]]:
        """Return ``(transformer, vae, text_encoder_1, text_encoder_2_or_None)``."""

    @abstractmethod
    def _encode_text(self, captions: list[str], ctx: TrainContext) -> tuple[Tensor, ...]:
        """Encode a batch of captions on ``ctx.device``; grad flows if an encoder is trained."""

    @abstractmethod
    def _flow_loss(self, latents: Tensor, text: tuple[Tensor, ...], ctx: TrainContext) -> Tensor:
        """Flow-matching loss for raw (unscaled) VAE latents ``[B, C, H, W]``."""

    # ------------------------------------------------------------------------ setup

    def _trained_text_parts(self) -> dict[str, nn.Module]:
        config = self._config
        parts: dict[str, nn.Module] = {}
        if config.text_encoder_1.train and self._text_encoder_1 is not None:
            parts["text_encoder_1"] = self._text_encoder_1
        if config.text_encoder_2.train and self._text_encoder_2 is not None:
            parts["text_encoder_2"] = self._text_encoder_2
        return parts

    def _lora_parts(self) -> dict[str, nn.Module]:
        assert self._transformer is not None
        return {"transformer": self._transformer, **self._trained_text_parts()}

    def _setup_model(self, ctx: TrainContext) -> list[dict]:
        config = ctx.config
        log = ctx.log
        device = ctx.device
        if ctx.weight_dtype != torch.bfloat16:
            raise ValueError("Flux training requires mixed_precision=bfloat16")

        log.info(
            "Loading %s components (quantization=%s)", self._model_label, self._quantization.value
        )
        transformer, vae, text_encoder_1, text_encoder_2 = self._load_components(ctx)
        self._guidance_embeds = bool(getattr(transformer.config, "guidance_embeds", False))
        for module in (transformer, vae, text_encoder_1, text_encoder_2):
            if module is not None:
                module.requires_grad_(False)

        def _lora_cfg(targets: str | list[str]):
            return build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=targets,  # type: ignore[arg-type]
            )

        transformer = get_peft_model(transformer, _lora_cfg(self._transformer_targets))
        param_groups: list[dict] = [
            {
                "params": [p for p in transformer.parameters() if p.requires_grad],
                "lr": config.resolve_learning_rate("unet"),
            }
        ]

        targets = self._text_encoder_targets
        if config.text_encoder_1.train:
            text_encoder_1 = get_peft_model(text_encoder_1, _lora_cfg(targets["text_encoder_1"]))
            param_groups.append(
                {
                    "params": [p for p in text_encoder_1.parameters() if p.requires_grad],
                    "lr": config.resolve_learning_rate("text_encoder_1"),
                }
            )
        if config.text_encoder_2.train and text_encoder_2 is not None:
            text_encoder_2 = get_peft_model(text_encoder_2, _lora_cfg(targets["text_encoder_2"]))
            param_groups.append(
                {
                    "params": [p for p in text_encoder_2.parameters() if p.requires_grad],
                    "lr": config.resolve_learning_rate("text_encoder_2"),
                }
            )

        self._transformer = transformer
        self._vae = vae
        self._text_encoder_1 = text_encoder_1
        self._text_encoder_2 = text_encoder_2

        if ctx.resume_state is not None:
            applied = apply_flux_resume_state(self._lora_parts(), ctx.resume_state.lora_state_dict)
            log.info("Restored %d LoRA tensors from resume state", applied)

        fp32_count = cast_trainable_params_to_fp32(*self._lora_parts().values())
        log.info("LoRA trainable params cast to fp32 (N=%d)", fp32_count)

        if config.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()

        transformer.to(device)
        if not config.cache_latents:
            vae.to(device)
        if not config.cache_text_encoder_outputs:
            for encoder in (text_encoder_1, text_encoder_2):
                if encoder is not None:
                    encoder.to(device)

        self._forward_transformer = transformer
        if config.torch_compile:
            log.info("Compiling transformer with torch.compile (inductor)...")
            self._forward_transformer = torch.compile(transformer, backend="inductor")
        return param_groups

    def _set_train_mode(self) -> None:
        assert self._transformer is not None
        self._transformer.train()
        for module in self._trained_text_parts().values():
            module.train()

    # ----------------------------------------------------------------------- caches

    def _encode_caption_cpu(self, caption: str, ctx: TrainContext) -> tuple[Tensor, ...]:
        return tuple(t.cpu() for t in self._encode_text([caption], ctx))

    def _build_caches(
        self,
        ctx: TrainContext,
        all_paths: list[Path],
        all_pairs: list[tuple[Path, str]],
        on_progress: Optional[CacheProgress],
    ) -> None:
        config = ctx.config
        log = ctx.log
        if config.cache_latents:
            assert self._vae is not None
            self._latent_cache = build_flux_latent_cache(
                all_paths,
                self._vae,
                ctx.device,
                config.cache_latents_to_disk,
                tag=self._tag,
                on_progress=on_progress,
                log=log,
            )
        if config.cache_text_encoder_outputs:
            if config.cache_text_encoder_outputs_to_disk:
                log.warning(
                    "Text embedding disk cache is enabled. Delete *_te.pt files next to images "
                    "after changing trigger_words or captions."
                )
            encoders = [e for e in (self._text_encoder_1, self._text_encoder_2) if e is not None]
            self._text_cache = build_flux_text_cache(
                all_pairs,
                lambda caption: self._encode_caption_cpu(caption, ctx),
                encoders,
                ctx.device,
                ctx.weight_dtype,
                config.cache_text_encoder_outputs_to_disk,
                tag=self._text_cache_tag,
                on_progress=on_progress,
                log=log,
            )

    # ------------------------------------------------------------------------- step

    def _batch_latents(self, batch: dict, ctx: TrainContext) -> Tensor:
        device = ctx.device
        if self._latent_cache is not None:
            return torch.stack([self._latent_cache[p] for p in batch["image_path"]]).to(
                device, dtype=torch.float32
            )
        assert self._vae is not None
        vae_dtype = next(self._vae.parameters()).dtype
        if "pixel_values" in batch:
            pixels = batch["pixel_values"]
        else:
            pixels = torch.stack(
                [_PIXEL_TRANSFORM(Image.open(p).convert("RGB")) for p in batch["image_path"]]
            )
        with torch.no_grad():
            latents = self._vae.encode(pixels.to(device, dtype=vae_dtype)).latent_dist.sample()
        return latents.float()

    def _batch_text(self, batch: dict, ctx: TrainContext) -> tuple[Tensor, ...]:
        captions: list[str] = batch["caption"]
        if self._text_cache is not None:
            count = len(next(iter(self._text_cache.values())))
            return tuple(
                torch.cat([self._text_cache[c][i] for c in captions], dim=0).to(
                    ctx.device, dtype=ctx.weight_dtype
                )
                for i in range(count)
            )
        return self._encode_text(captions, ctx)

    def _compute_loss(self, batch: dict, ctx: TrainContext) -> Tensor:
        latents = self._batch_latents(batch, ctx)
        text = self._batch_text(batch, ctx)
        return self._flow_loss(latents, text, ctx)

    # ----------------------------------------------------------------------- export

    def _export_weights(self, path: Path) -> None:
        config = self._config
        state = export_flux_lora_state_dict(
            self._lora_parts(), rank=config.lora_rank, alpha=config.lora_alpha
        )
        write_flux_lora(state, path, config.output_format.value)

    def _collect_lora_state(self) -> dict[str, Tensor]:
        return collect_flux_resume_state(self._lora_parts())

    @staticmethod
    def _validate_config(config: TrainConfig) -> None:
        config.validate_gpu()
        config.validate_model_files()
