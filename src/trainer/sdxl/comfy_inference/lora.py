"""Apply LoRA through vendored Comfy ModelPatcher."""

from __future__ import annotations

from typing import Any

from src.trainer.sdxl.comfy_inference.bootstrap import ensure_vendored_comfy
from src.trainer.sdxl.comfy_inference.types import ComfyInferenceStack


def apply_comfy_lora(stack: ComfyInferenceStack, *, lora_weight: float) -> tuple[Any, Any]:
    if stack.lora_state is None:
        return stack.model, stack.clip

    ensure_vendored_comfy()
    import comfy.sd as comfy_sd

    strength_clip = lora_weight if (stack.lora_apply_te1 or stack.lora_apply_te2) else 0.0
    return comfy_sd.load_lora_for_models(
        stack._base_model,
        stack._base_clip,
        stack.lora_state,
        lora_weight,
        strength_clip,
    )
