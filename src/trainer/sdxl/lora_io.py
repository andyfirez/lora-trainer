"""Load and apply LoRA state dicts for SDXL training and sampling."""

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from src.trainer.config import TrainConfig
from src.trainer.sdxl.lora_export import apply_kohya_state_dict, detect_lora_format


def load_lora_file(lora_path: Path) -> dict[str, Any]:
    if lora_path.suffix == ".safetensors":
        return dict(load_file(str(lora_path), device="cpu"))
    data = torch.load(lora_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported LoRA file format: {lora_path}")
    return data


def apply_lora_state_to_module(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for name, param in module.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        key = f"{prefix}{name.replace('.', '_')}"
        value = state_dict.get(key)
        if value is None:
            continue
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(
                f"LoRA weight shape mismatch for {key}: "
                f"checkpoint {tuple(value.shape)} vs model {tuple(param.shape)}"
            )
        with torch.no_grad():
            param.copy_(value.to(dtype=param.dtype, device=param.device))


def apply_lora_state_dict(
    state_dict: dict[str, Any],
    *,
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: TrainConfig,
) -> None:
    if detect_lora_format(state_dict) == "kohya":
        apply_kohya_state_dict(
            state_dict,
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            config=config,
        )
        return

    apply_lora_state_to_module(unet, state_dict, prefix="lora_unet_")
    if config.text_encoder_1.train:
        apply_lora_state_to_module(text_encoder_1, state_dict, prefix="lora_te1_")
    if config.text_encoder_2.train:
        apply_lora_state_to_module(text_encoder_2, state_dict, prefix="lora_te2_")
