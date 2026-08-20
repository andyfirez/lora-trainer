"""Shared torch dtype mapping for SDXL training and inference."""

from __future__ import annotations

import torch

from src.trainer.config import VaeDtype, WeightDtype

_TORCH_DTYPES: dict[WeightDtype | VaeDtype, torch.dtype] = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
    VaeDtype.FLOAT_32: torch.float32,
    VaeDtype.FLOAT_16: torch.float16,
    VaeDtype.BFLOAT_16: torch.bfloat16,
}


def weight_dtype_to_torch(dtype: WeightDtype | VaeDtype) -> torch.dtype:
    return _TORCH_DTYPES[dtype]
