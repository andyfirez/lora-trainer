"""Smoke tests for SDXL LoRA PEFT target module lists."""

from src.trainer.sdxl.lora_targets import (
    SDXL_TE_LORA_TARGET_MODULES,
    SDXL_UNET_LORA_TARGET_MODULES,
)


def test_lora_target_modules_are_unique_and_non_empty() -> None:
    assert SDXL_UNET_LORA_TARGET_MODULES
    assert SDXL_TE_LORA_TARGET_MODULES
    assert len(SDXL_UNET_LORA_TARGET_MODULES) == len(set(SDXL_UNET_LORA_TARGET_MODULES))
    assert len(SDXL_TE_LORA_TARGET_MODULES) == len(set(SDXL_TE_LORA_TARGET_MODULES))
