"""SDXL LoRA trainer using diffusers + peft."""

import contextlib
import logging
from pathlib import Path
from typing import Optional

import torch
from peft import get_peft_model
from torch import Tensor

from src.storage.config_paths import resolve_config_base_model
from src.trainer.attention import configure_unet_attention
from src.trainer.base_trainer import (
    DTYPE_MAP as _DTYPE_MAP,
)
from src.trainer.base_trainer import (
    BaseLoRATrainer,
    CacheProgress,
    TrainContext,
    TrainingCancelledAfterSave,
    TrainingCancelledDuringCache,
)
from src.trainer.sdxl.latent_cache import build_latent_cache
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config
from src.trainer.sdxl.lora_persistence import (
    collect_lora_state_dict,
    export_lora_weights,
    load_lora_state_dict,
)
from src.trainer.sdxl.lora_targets import (
    SDXL_TE_LORA_TARGET_MODULES,
    SDXL_UNET_LORA_TARGET_MODULES,
)
from src.trainer.sdxl.loss import apply_noise_offset, min_snr_weight
from src.trainer.sdxl.mixed_precision import cast_trainable_params_to_fp32
from src.trainer.sdxl.model_loader import load_sdxl_components
from src.trainer.sdxl.prompt_encoding import select_clip_hidden_state
from src.trainer.sdxl.te_cache import build_te_cache

logger = logging.getLogger(__name__)

__all__ = [
    "SDXLLoRATrainer",
    "TrainingCancelledAfterSave",
    "TrainingCancelledDuringCache",
    "_DTYPE_MAP",
]


class SDXLLoRATrainer(BaseLoRATrainer):
    _model_label = "SDXL"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._unet = None
        self._text_encoder_1 = None
        self._text_encoder_2 = None
        self._vae = None
        self._tokenizer_1 = None
        self._tokenizer_2 = None
        self._noise_scheduler = None
        self._latent_cache: Optional[dict[str, Tensor]] = None
        self._te_cache: Optional[dict[str, tuple[Tensor, Tensor]]] = None

    # ------------------------------------------------------------------ setup

    def _setup_model(self, ctx: TrainContext) -> list[dict]:
        config = ctx.config
        log = ctx.log
        device = ctx.device
        resolved_base_model = resolve_config_base_model(config.base_model_name)
        log.info("Loading SDXL pipeline from %s", resolved_base_model)

        components = load_sdxl_components(
            resolved_base_model,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
            vae_dtype=ctx.vae_dtype,
        )
        self._tokenizer_1 = components.tokenizer_1
        self._tokenizer_2 = components.tokenizer_2
        self._noise_scheduler = components.noise_scheduler
        text_encoder_1 = components.text_encoder_1
        text_encoder_2 = components.text_encoder_2
        vae = components.vae
        unet = components.unet

        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)

        unet_lora_config = build_sdxl_lora_config(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=SDXL_UNET_LORA_TARGET_MODULES,
        )
        unet = get_peft_model(unet, unet_lora_config)
        param_groups: list[dict] = [
            {"params": list(unet.parameters()), "lr": config.resolve_learning_rate("unet")}
        ]

        if config.text_encoder_1.train:
            te1_lora_config = build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_TE_LORA_TARGET_MODULES,
            )
            text_encoder_1 = get_peft_model(text_encoder_1, te1_lora_config)
            param_groups.append(
                {
                    "params": list(text_encoder_1.parameters()),
                    "lr": config.resolve_learning_rate("text_encoder_1"),
                }
            )

        if config.text_encoder_2.train:
            te2_lora_config = build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_TE_LORA_TARGET_MODULES,
            )
            text_encoder_2 = get_peft_model(text_encoder_2, te2_lora_config)
            param_groups.append(
                {
                    "params": list(text_encoder_2.parameters()),
                    "lr": config.resolve_learning_rate("text_encoder_2"),
                }
            )

        if ctx.resume_state is not None:
            load_lora_state_dict(
                ctx.resume_state.lora_state_dict,
                unet=unet,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                config=config,
            )

        fp32_cast_count = cast_trainable_params_to_fp32(unet, text_encoder_1, text_encoder_2)
        log.info("LoRA trainable params cast to fp32 (N=%d)", fp32_cast_count)

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        configure_unet_attention(unet, ctx.attention_mechanism, log)

        # UNet always on GPU.
        unet = unet.to(device)

        # VAE: stays on CPU when latents will be cached (build_latent_cache manages device).
        if not config.cache_latents:
            vae = vae.to(device)

        # TEs: to GPU only if they will be called during training (i.e., not cached).
        if not config.cache_text_encoder_outputs:
            text_encoder_1 = text_encoder_1.to(device)
            text_encoder_2 = text_encoder_2.to(device)

        if config.torch_compile:
            log.info("Compiling UNet with torch.compile (inductor)...")
            unet = torch.compile(unet, backend="inductor")

        self._unet = unet
        self._text_encoder_1 = text_encoder_1
        self._text_encoder_2 = text_encoder_2
        self._vae = vae
        return param_groups

    def _set_train_mode(self) -> None:
        config = self._config
        self._unet.train()
        if config.text_encoder_1.train:
            self._text_encoder_1.train()
        if config.text_encoder_2.train:
            self._text_encoder_2.train()

    # ----------------------------------------------------------------- caches

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
            log.info("Building latent cache...")
            self._latent_cache = build_latent_cache(
                all_paths,
                self._vae,
                ctx.device,
                config.cache_latents_to_disk,
                on_progress=on_progress,
                log=log,
            )

        if config.cache_text_encoder_outputs_to_disk:
            log.warning(
                "Text encoder disk cache is enabled. Delete *_te.npz files next to images "
                "after changing trigger_words, captions, or clip_skip."
            )

        if config.cache_text_encoder_outputs:
            log.info("Building text encoder cache...")
            self._te_cache = build_te_cache(
                all_pairs,
                self._tokenizer_1,
                self._tokenizer_2,
                self._text_encoder_1,
                self._text_encoder_2,
                ctx.device,
                ctx.weight_dtype,
                config.clip_skip,
                config.cache_text_encoder_outputs_to_disk,
                on_progress=on_progress,
                log=log,
            )

    # ------------------------------------------------------------------- step

    def _compute_loss(self, batch: dict, ctx: TrainContext) -> Tensor:
        config = ctx.config
        device = ctx.device
        weight_dtype = ctx.weight_dtype
        vae = self._vae
        unet = self._unet
        noise_scheduler = self._noise_scheduler
        captions: list[str] = batch["caption"]

        # --- Latents ---
        if self._latent_cache is not None:
            image_paths: list[str] = batch["image_path"]
            latents = torch.stack(
                [self._latent_cache[p].to(device, dtype=weight_dtype) for p in image_paths]
            )
        else:
            pixel_values = batch["pixel_values"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

        # --- Noise + timesteps ---
        with torch.no_grad():
            noise = torch.randn_like(latents)
            noise = apply_noise_offset(latents, noise, config.noise_offset)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            add_time_ids = batch["add_time_ids"].to(dtype=weight_dtype, device=device)

        # --- Text embeddings ---
        if self._te_cache is not None:
            prompt_embeds = torch.cat(
                [self._te_cache[c][0].to(device, dtype=weight_dtype) for c in captions], dim=0
            )
            pooled_prompt_embeds = torch.cat(
                [self._te_cache[c][1].to(device, dtype=weight_dtype) for c in captions], dim=0
            )
        else:
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
                captions,
                self._tokenizer_1,
                self._tokenizer_2,
                self._text_encoder_1,
                self._text_encoder_2,
                device,
                weight_dtype,
                config.clip_skip,
                train_te1=config.text_encoder_1.train,
                train_te2=config.text_encoder_2.train,
            )

        # --- UNet forward + loss ---
        with torch.autocast(device_type=device.type, dtype=weight_dtype):
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample

        target = (
            noise
            if noise_scheduler.config.prediction_type == "epsilon"
            else noise_scheduler.get_velocity(latents, noise, timesteps)
        )
        per_sample_loss = torch.nn.functional.mse_loss(
            model_pred.float(),
            target.float(),
            reduction="none",
        ).mean(dim=(1, 2, 3))
        if config.min_snr_gamma > 0:
            v_prediction = noise_scheduler.config.prediction_type == "v_prediction"
            snr_weights = min_snr_weight(
                timesteps,
                noise_scheduler.alphas_cumprod,
                config.min_snr_gamma,
                v_prediction=v_prediction,
            )
            per_sample_loss = per_sample_loss * snr_weights.to(device=per_sample_loss.device)
        return per_sample_loss.mean()

    def _encode_prompt(
        self,
        captions: list[str],
        tokenizer_1,
        tokenizer_2,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        clip_skip: int,
        train_te1: bool,
        train_te2: bool,
    ) -> tuple[Tensor, Tensor]:
        tokens_1 = tokenizer_1(
            captions,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Use no_grad for frozen encoders; enable_grad for trained ones so LoRA receives gradients.
        te1_ctx = contextlib.nullcontext() if train_te1 else torch.no_grad()
        te2_ctx = contextlib.nullcontext() if train_te2 else torch.no_grad()

        with te1_ctx:
            enc1_out = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_1 = select_clip_hidden_state(enc1_out.hidden_states, clip_skip).to(dtype=dtype)

        with te2_ctx:
            enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_2 = select_clip_hidden_state(enc2_out.hidden_states, clip_skip).to(dtype=dtype)
            pooled_prompt_embeds = enc2_out[0].to(dtype=dtype)

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    # ----------------------------------------------------------------- export

    def _export_weights(self, path: Path) -> None:
        export_lora_weights(
            self._unet, self._text_encoder_1, self._text_encoder_2, self._config, path
        )

    def _collect_lora_state(self) -> dict[str, Tensor]:
        return collect_lora_state_dict(
            self._unet, self._text_encoder_1, self._text_encoder_2, self._config
        )
