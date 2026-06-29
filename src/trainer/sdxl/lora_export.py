"""Convert PEFT LoRA weights to/from kohya-compatible safetensors format."""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from src.trainer.config import TrainConfig

_PEFT_PREFIX = "base_model.model."
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"


def detect_lora_format(state_dict: dict[str, Any]) -> Literal["kohya", "legacy"]:
    if any(".lora_down.weight" in key for key in state_dict):
        return "kohya"
    return "legacy"


@dataclass(frozen=True)
class KohyaLoRAMetadata:
    rank: int
    alpha: float
    train_te1: bool
    train_te2: bool


def infer_kohya_lora_metadata(state_dict: dict[str, Any]) -> KohyaLoRAMetadata:
    if detect_lora_format(state_dict) != "kohya":
        raise ValueError("Cannot infer LoRA metadata from non-kohya state dict")

    ranks: set[int] = set()
    alphas: list[float] = []
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        if key.endswith(".lora_down.weight"):
            ranks.add(int(value.shape[0]))
        elif key.endswith(".alpha"):
            alphas.append(float(value.item()))

    if not ranks:
        raise ValueError("No LoRA weights found in state dict")

    if len(ranks) > 1:
        raise ValueError(f"Inconsistent LoRA ranks in checkpoint: {sorted(ranks)}")

    rank = ranks.pop()
    alpha = alphas[0] if alphas else float(rank)
    train_te1 = any(key.startswith("lora_te1_") for key in state_dict)
    train_te2 = any(key.startswith("lora_te2_") for key in state_dict)
    return KohyaLoRAMetadata(rank=rank, alpha=alpha, train_te1=train_te1, train_te2=train_te2)


def apply_lora_metadata_to_config(config: TrainConfig, state_dict: dict[str, Any]) -> TrainConfig:
    metadata = infer_kohya_lora_metadata(state_dict)
    return config.model_copy(
        update={
            "lora_rank": metadata.rank,
            "lora_alpha": metadata.alpha,
            "text_encoder_1": config.text_encoder_1.model_copy(update={"train": metadata.train_te1}),
            "text_encoder_2": config.text_encoder_2.model_copy(update={"train": metadata.train_te2}),
        }
    )


def _peft_param_to_kohya_keys(name: str, prefix: str) -> tuple[str, str] | None:
    if not name.startswith(_PEFT_PREFIX):
        return None
    if name.endswith(_LORA_A_SUFFIX):
        module_path = name[len(_PEFT_PREFIX) : -len(_LORA_A_SUFFIX)]
        weight_suffix = "lora_down.weight"
    elif name.endswith(_LORA_B_SUFFIX):
        module_path = name[len(_PEFT_PREFIX) : -len(_LORA_B_SUFFIX)]
        weight_suffix = "lora_up.weight"
    else:
        return None
    kohya_base = f"{prefix}{module_path.replace('.', '_')}"
    return kohya_base, weight_suffix


def _collect_module_kohya_state(
    module: torch.nn.Module,
    *,
    prefix: str,
    lora_alpha: float,
) -> dict[str, Tensor]:
    state_dict: dict[str, Tensor] = {}
    alpha_bases: set[str] = set()

    for name, param in module.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        keys = _peft_param_to_kohya_keys(name, prefix)
        if keys is None:
            continue
        kohya_base, weight_suffix = keys
        state_dict[f"{kohya_base}.{weight_suffix}"] = param.detach().cpu()
        alpha_bases.add(kohya_base)

    alpha_tensor = torch.tensor(lora_alpha)
    for kohya_base in alpha_bases:
        state_dict[f"{kohya_base}.alpha"] = alpha_tensor.clone()

    return state_dict


def export_kohya_state_dict(
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: TrainConfig,
) -> dict[str, Tensor]:
    state_dict = _collect_module_kohya_state(
        unet,
        prefix="lora_unet_",
        lora_alpha=config.lora_alpha,
    )
    if config.text_encoder_1.train:
        state_dict.update(
            _collect_module_kohya_state(
                text_encoder_1,
                prefix="lora_te1_",
                lora_alpha=config.lora_alpha,
            )
        )
    if config.text_encoder_2.train:
        state_dict.update(
            _collect_module_kohya_state(
                text_encoder_2,
                prefix="lora_te2_",
                lora_alpha=config.lora_alpha,
            )
        )
    return state_dict


def _apply_kohya_state_to_module(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for name, param in module.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        keys = _peft_param_to_kohya_keys(name, prefix)
        if keys is None:
            continue
        kohya_base, weight_suffix = keys
        value = state_dict.get(f"{kohya_base}.{weight_suffix}")
        if value is None:
            continue
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(
                f"LoRA weight shape mismatch for {kohya_base}.{weight_suffix}: "
                f"checkpoint {tuple(value.shape)} vs model {tuple(param.shape)}"
            )
        with torch.no_grad():
            param.copy_(value.to(dtype=param.dtype, device=param.device))


def apply_kohya_state_dict(
    state_dict: dict[str, Any],
    *,
    unet: torch.nn.Module,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    config: TrainConfig,
) -> None:
    _apply_kohya_state_to_module(unet, state_dict, prefix="lora_unet_")
    if config.text_encoder_1.train:
        _apply_kohya_state_to_module(text_encoder_1, state_dict, prefix="lora_te1_")
    if config.text_encoder_2.train:
        _apply_kohya_state_to_module(text_encoder_2, state_dict, prefix="lora_te2_")
