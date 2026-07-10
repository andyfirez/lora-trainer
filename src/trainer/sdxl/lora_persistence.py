"""LoRA checkpoint export and state-dict persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.trainer.config import TrainConfig
from src.trainer.sdxl.lora_export import export_kohya_state_dict
from src.trainer.sdxl.lora_io import apply_lora_state_dict, apply_lora_state_to_module


def export_lora_weights(
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: TrainConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file  # noqa: PLC0415

    state_dict = export_kohya_state_dict(unet, text_encoder_1, text_encoder_2, config)
    if config.output_format.value == "safetensors":
        save_file(state_dict, str(path))
    else:
        torch.save(state_dict, str(path))


def collect_lora_state_dict(
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: TrainConfig,
) -> dict[str, Tensor]:
    return export_kohya_state_dict(unet, text_encoder_1, text_encoder_2, config)


def load_lora_state_dict(
    state_dict: dict[str, Any],
    *,
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: TrainConfig,
) -> None:
    apply_lora_state_dict(
        state_dict,
        unet=unet,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        config=config,
    )


def apply_lora_state_to_module_prefix(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    prefix: str,
) -> None:
    apply_lora_state_to_module(module, state_dict, prefix=prefix)
