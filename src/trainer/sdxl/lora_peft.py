"""Shared PEFT LoRA config builder for SDXL training and sampling."""

from peft import LoraConfig


def build_sdxl_lora_config(
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> LoraConfig:
    """Build Kohya-compatible LoRA config: Kaiming lora_A, zero lora_B."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights=True,
        target_modules=target_modules,
    )
