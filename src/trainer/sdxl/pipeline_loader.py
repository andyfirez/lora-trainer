"""Shared SDXL pipeline loading for training-adjacent inference (sampling/sweep)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.storage.config_paths import resolve_config_base_model
from src.trainer.attention import configure_unet_attention
from src.trainer.config import TrainConfig
from src.gpu.runtime import setup_cuda_runtime
from src.trainer.sdxl.dtypes import weight_dtype_to_torch
from src.trainer.sdxl.lora_export import apply_lora_metadata_to_config
from src.trainer.sdxl.lora_io import apply_lora_state_dict, load_lora_file
from src.trainer.sdxl.lora_peft import attach_sdxl_lora_adapters
from src.trainer.sdxl.model_loader import load_sdxl_components, resolve_vae_dtype

logger = logging.getLogger(__name__)


@dataclass
class SamplingStack:
    device: torch.device
    tokenizer_1: Any
    tokenizer_2: Any
    noise_scheduler: Any
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module
    vae: torch.nn.Module
    unet: torch.nn.Module


class SDXLPipelineLoader:
    """Load SDXL components and optional LoRA weights without sweep orchestration."""

    def __init__(
        self,
        base_config: TrainConfig,
        *,
        log: logging.Logger | None = None,
    ) -> None:
        self._base_config = base_config
        self._log = log or logger

    def load_stack_for_combo(
        self,
        *,
        base_model: str,
        lora_path: Path | None,
        combo_params: dict[str, Any],
    ) -> tuple[SamplingStack, TrainConfig, bool]:
        config = self._base_config.model_copy(update={"base_model_name": base_model})
        if lora_path is not None:
            self._log.info("Reading LoRA file: %s", lora_path)
            load_started = time.perf_counter()
            state_dict = load_lora_file(lora_path)
            self._log.info("LoRA file read in %.1fs", time.perf_counter() - load_started)
            lora_config = apply_lora_metadata_to_config(config, state_dict)
            self._log.info(
                "LoRA metadata: rank=%d alpha=%.1f te1=%s te2=%s",
                lora_config.lora_rank,
                lora_config.lora_alpha,
                lora_config.text_encoder_1.train,
                lora_config.text_encoder_2.train,
            )
            stack = self.load_stack(lora_config, enable_lora=True)
            self._log.info("Applying LoRA weights to pipeline...")
            apply_started = time.perf_counter()
            apply_lora_state_dict(
                state_dict,
                unet=stack.unet,
                text_encoder_1=stack.text_encoder_1,
                text_encoder_2=stack.text_encoder_2,
                config=lora_config,
            )
            self._log.info("LoRA weights applied in %.1fs", time.perf_counter() - apply_started)
            return stack, lora_config, True

        self._log.info("Loading base model pipeline (no LoRA)")
        stack = self.load_stack(config, enable_lora=False)
        return stack, config, False

    def load_stack(self, config: TrainConfig, *, enable_lora: bool) -> SamplingStack:
        runtime = setup_cuda_runtime(config)
        device = runtime.device
        gpu = runtime.gpu
        vae_dtype = resolve_vae_dtype(gpu.vae_dtype)
        resolved_base_model = resolve_config_base_model(config.base_model_name)
        self._log.info(
            "Loading SDXL components from %s (lora=%s, attention=%s)...",
            resolved_base_model,
            enable_lora,
            gpu.attention_mechanism,
        )
        load_started = time.perf_counter()
        components = load_sdxl_components(
            resolved_base_model,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
            vae_dtype=gpu.vae_dtype,
        )
        self._log.info("SDXL components loaded from disk in %.1fs", time.perf_counter() - load_started)

        vae = components.vae
        text_encoder_1 = components.text_encoder_1
        text_encoder_2 = components.text_encoder_2
        unet = components.unet

        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)

        if enable_lora:
            self._log.info("Attaching LoRA adapters (rank=%d)...", config.lora_rank)
            attachment = attach_sdxl_lora_adapters(
                unet,
                text_encoder_1,
                text_encoder_2,
                config,
                enable_lora=True,
            )
            unet = attachment.unet
            text_encoder_1 = attachment.text_encoder_1
            text_encoder_2 = attachment.text_encoder_2

        self._log.info("Moving SDXL models to GPU...")
        gpu_started = time.perf_counter()
        unet = unet.to(device=device, dtype=weight_dtype_to_torch(config.unet.weight_dtype))
        text_encoder_1 = text_encoder_1.to(
            device=device,
            dtype=weight_dtype_to_torch(config.text_encoder_1.weight_dtype),
        )
        text_encoder_2 = text_encoder_2.to(
            device=device,
            dtype=weight_dtype_to_torch(config.text_encoder_2.weight_dtype),
        )
        vae = vae.to(device=device, dtype=vae_dtype)
        self._log.info("GPU transfer finished in %.1fs", time.perf_counter() - gpu_started)
        configure_unet_attention(unet, gpu.attention_mechanism, self._log)
        self._log.info("Pipeline ready for sampling")

        return SamplingStack(
            device=device,
            tokenizer_1=components.tokenizer_1,
            tokenizer_2=components.tokenizer_2,
            noise_scheduler=components.noise_scheduler,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            unet=unet,
        )
