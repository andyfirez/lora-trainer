"""PEFT LoRA targets, ComfyUI/diffusers-style export and resume state for Flux models.

Export format: diffusers LoRA keys (``transformer.<module>.lora_A.weight`` /
``lora_B.weight``, ``text_encoder.*`` / ``text_encoder_2.*`` for trained encoders), which
ComfyUI, Forge and diffusers all read. The ``alpha / rank`` scale is folded into
``lora_B`` so no ``.alpha`` tensors are required.

Resume state keeps raw PEFT parameters (unscaled) keyed by ``<part>|<peft param name>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

_PEFT_PREFIX = "base_model.model."
_A_SUFFIX = ".lora_A.default.weight"
_B_SUFFIX = ".lora_B.default.weight"

_DOUBLE_ATTN = r"attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)"

# Regexes are full-matched by PEFT against module names of the wrapped model.
FLUX1_TRANSFORMER_TARGETS = (
    rf"transformer_blocks\.\d+\.({_DOUBLE_ATTN}|ff\.net\.0\.proj|ff\.net\.2"
    r"|ff_context\.net\.0\.proj|ff_context\.net\.2)"
    r"|single_transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v)|proj_mlp|proj_out)"
)

FLUX2_TRANSFORMER_TARGETS = (
    rf"transformer_blocks\.\d+\.({_DOUBLE_ATTN}|ff\.linear_in|ff\.linear_out"
    r"|ff_context\.linear_in|ff_context\.linear_out)"
    r"|single_transformer_blocks\.\d+\.attn\.(to_qkv_mlp_proj|to_out)"
)

CLIP_TARGETS: list[str] = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
T5_TARGETS: list[str] = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
QWEN3_TARGETS: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Export prefix per trained part (matches diffusers' save_lora_weights naming).
PART_EXPORT_PREFIX = {
    "transformer": "transformer",
    "text_encoder_1": "text_encoder",
    "text_encoder_2": "text_encoder_2",
}


def _lora_param_items(module: nn.Module):
    for name, param in module.named_parameters():
        if "lora_" in name and param.requires_grad:
            yield name, param


def export_flux_lora_state_dict(
    parts: dict[str, nn.Module],
    *,
    rank: int,
    alpha: float,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Tensor]:
    """Build the inference-format state dict. ``parts`` maps part name -> PEFT-wrapped module."""
    scale = float(alpha) / float(rank)
    state: dict[str, Tensor] = {}
    for part, module in parts.items():
        export_prefix = PART_EXPORT_PREFIX[part]
        for name, param in _lora_param_items(module):
            if not name.startswith(_PEFT_PREFIX):
                continue
            if name.endswith(_A_SUFFIX):
                path = name[len(_PEFT_PREFIX): -len(_A_SUFFIX)]
                tensor = param.detach().float()
                suffix = "lora_A.weight"
            elif name.endswith(_B_SUFFIX):
                path = name[len(_PEFT_PREFIX): -len(_B_SUFFIX)]
                tensor = param.detach().float() * scale
                suffix = "lora_B.weight"
            else:
                continue
            state[f"{export_prefix}.{path}.{suffix}"] = tensor.to(dtype).cpu().contiguous()
    return state


def write_flux_lora(state_dict: dict[str, Tensor], path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "safetensors":
        from safetensors.torch import save_file  # noqa: PLC0415

        save_file(state_dict, str(path), metadata={"format": "pt"})
    else:
        torch.save(state_dict, str(path))


def collect_flux_resume_state(parts: dict[str, nn.Module]) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for part, module in parts.items():
        for name, param in _lora_param_items(module):
            state[f"{part}|{name}"] = param.detach().cpu().clone()
    return state


def apply_flux_resume_state(parts: dict[str, nn.Module], state: dict[str, Any]) -> int:
    """Copy saved LoRA tensors into the matching parameters; return how many were applied."""
    applied = 0
    for part, module in parts.items():
        for name, param in _lora_param_items(module):
            value = state.get(f"{part}|{name}")
            if value is None:
                continue
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(
                    f"LoRA weight shape mismatch for {part}|{name}: "
                    f"checkpoint {tuple(value.shape)} vs model {tuple(param.shape)}"
                )
            with torch.no_grad():
                param.copy_(value.to(dtype=param.dtype, device=param.device))
            applied += 1
    if applied == 0:
        raise ValueError("Resume state does not contain any LoRA weights for this model")
    return applied
