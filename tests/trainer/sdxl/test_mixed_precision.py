"""Tests for mixed-precision helpers (fp32 LoRA + GradScaler)."""

import torch
import torch.nn as nn
from peft import get_peft_model
from src.trainer.config import WeightDtype
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config
from src.trainer.sdxl.mixed_precision import (
    cast_trainable_params_to_fp32,
    create_grad_scaler,
)


class _LinearTargetModule(nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.to_q = nn.Linear(in_features, out_features, bias=False)


def _build_fp16_peft_model() -> nn.Module:
    base = _LinearTargetModule()
    base = base.to(dtype=torch.float16)
    base.requires_grad_(False)
    model = get_peft_model(
        base,
        build_sdxl_lora_config(
            rank=4,
            alpha=4.0,
            dropout=0.0,
            target_modules=["to_q"],
        ),
    )
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)
    return model


def test_cast_trainable_params_to_fp32_leaves_frozen_base_fp16() -> None:
    model = _build_fp16_peft_model()
    updated = cast_trainable_params_to_fp32(model)

    assert updated > 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.dtype == torch.float32
            assert param.requires_grad
        else:
            assert param.dtype == torch.float16
            assert not param.requires_grad


def test_create_grad_scaler_enabled_for_float16() -> None:
    scaler = create_grad_scaler(WeightDtype.FLOAT_16)
    assert scaler is not None


def test_create_grad_scaler_disabled_for_bfloat16_and_float32() -> None:
    assert create_grad_scaler(WeightDtype.BFLOAT_16) is None
    assert create_grad_scaler(WeightDtype.FLOAT_32) is None
