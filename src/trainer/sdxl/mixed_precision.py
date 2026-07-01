"""Mixed-precision helpers for SDXL LoRA training (Kohya-compatible fp16 stack)."""

import torch
from torch.amp import GradScaler

from src.trainer.config import WeightDtype


def cast_trainable_params_to_fp32(*modules: torch.nn.Module) -> int:
    """Cast requires_grad parameters to float32; return count updated."""
    updated = 0
    for module in modules:
        for param in module.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                updated += 1
    return updated


def create_grad_scaler(mixed_precision: WeightDtype) -> GradScaler | None:
    """Return GradScaler for float16 mixed precision; None for bf16/float32."""
    if mixed_precision == WeightDtype.FLOAT_16:
        return GradScaler("cuda")
    return None
