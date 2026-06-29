"""Shared SDXL LoRA PEFT target module names (Kohya Standard scope)."""

SDXL_UNET_LORA_TARGET_MODULES: list[str] = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]

SDXL_TE_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]
