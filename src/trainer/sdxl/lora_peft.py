"""Shared PEFT LoRA config builder for SDXL training and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from peft import LoraConfig, get_peft_model

from src.trainer.sdxl.lora_targets import (
    SDXL_TE_LORA_TARGET_MODULES,
    SDXL_UNET_LORA_TARGET_MODULES,
)


@dataclass(frozen=True)
class SdxlLoraAttachment:
    unet: torch.nn.Module
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module
    param_groups: list[dict] | None = None


def build_sdxl_lora_config(
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> LoraConfig:
    """Build Kohya-compatible LoRA config: Kaiming lora_A, zero lora_B."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights=True,
        target_modules=target_modules,
    )


class _SdxlLoraAttachConfig(Protocol):
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    unet: object
    text_encoder_1: object
    text_encoder_2: object


def attach_sdxl_lora_adapters(
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: _SdxlLoraAttachConfig,
    *,
    enable_lora: bool,
    for_training: bool = False,
) -> SdxlLoraAttachment:
    """Attach PEFT LoRA adapters to SDXL UNet and optional text encoders."""
    param_groups: list[dict] | None = [] if for_training else None

    if enable_lora or for_training:
        unet = get_peft_model(
            unet,
            build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_UNET_LORA_TARGET_MODULES,
            ),
        )
        if for_training:
            param_groups.append({"params": list(unet.parameters()), "lr": config.unet.learning_rate})

    if (enable_lora or for_training) and config.text_encoder_1.train:
        text_encoder_1 = get_peft_model(
            text_encoder_1,
            build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_TE_LORA_TARGET_MODULES,
            ),
        )
        if for_training:
            param_groups.append(
                {
                    "params": list(text_encoder_1.parameters()),
                    "lr": config.text_encoder_1.learning_rate,
                }
            )

    if (enable_lora or for_training) and config.text_encoder_2.train:
        text_encoder_2 = get_peft_model(
            text_encoder_2,
            build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_TE_LORA_TARGET_MODULES,
            ),
        )
        if for_training:
            param_groups.append(
                {
                    "params": list(text_encoder_2.parameters()),
                    "lr": config.text_encoder_2.learning_rate,
                }
            )

    return SdxlLoraAttachment(
        unet=unet,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        param_groups=param_groups,
    )
