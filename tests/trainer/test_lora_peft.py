"""Tests for Kohya-compatible PEFT LoRA initialization."""

import torch
import torch.nn as nn
from peft import get_peft_model
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config


class _LinearTargetModule(nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.to_q = nn.Linear(in_features, out_features, bias=False)


def test_build_sdxl_lora_config_uses_kohya_init() -> None:
    config = build_sdxl_lora_config(
        rank=16,
        alpha=16.0,
        dropout=0.0,
        target_modules=["to_q"],
    )
    assert config.init_lora_weights is True
    assert config.r == 16
    assert config.lora_alpha == 16.0


def test_kohya_init_zero_lora_b_on_unet_linear() -> None:
    x = torch.randn(1, 8)
    base = _LinearTargetModule()
    with torch.no_grad():
        base_output = base.to_q(x)

    model = get_peft_model(
        base,
        build_sdxl_lora_config(
            rank=4,
            alpha=4.0,
            dropout=0.0,
            target_modules=["to_q"],
        ),
    )

    lora_a_weights: list[torch.Tensor] = []
    lora_b_weights: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if "lora_A.default.weight" in name:
            lora_a_weights.append(param)
        elif "lora_B.default.weight" in name:
            lora_b_weights.append(param)

    assert lora_a_weights
    assert lora_b_weights
    for weight in lora_b_weights:
        assert torch.all(weight == 0).item()
    for weight in lora_a_weights:
        assert torch.any(weight != 0).item()

    with torch.no_grad():
        peft_output = model.base_model.model.to_q(x)
    assert torch.allclose(base_output, peft_output, atol=1e-6)
