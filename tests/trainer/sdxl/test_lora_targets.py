"""Tests for SDXL LoRA PEFT target module names."""

from src.trainer.sdxl.lora_targets import (
    SDXL_TE_LORA_TARGET_MODULES,
    SDXL_UNET_LORA_TARGET_MODULES,
)


def test_unet_target_modules_include_ff_layers() -> None:
    assert "to_k" in SDXL_UNET_LORA_TARGET_MODULES
    assert "ff.net.0.proj" in SDXL_UNET_LORA_TARGET_MODULES
    assert "ff.net.2" in SDXL_UNET_LORA_TARGET_MODULES


def test_te_target_modules_include_projection_layers() -> None:
    assert SDXL_TE_LORA_TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "out_proj"]
