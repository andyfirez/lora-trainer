"""In-repo ComfyUI 0.27.0-derived SDXL inference stack (GPL-3 vendored)."""

from src.trainer.sdxl.comfy_inference.loader import load_comfy_sdxl_stack
from src.trainer.sdxl.comfy_inference.runner import run_comfy_inference_sampling
from src.trainer.sdxl.comfy_inference.types import ComfyInferenceStack

__all__ = [
    "ComfyInferenceStack",
    "load_comfy_sdxl_stack",
    "run_comfy_inference_sampling",
]
