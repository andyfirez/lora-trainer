"""Validate dtype and attention settings against the local GPU."""

import torch
from diffusers.utils.import_utils import is_xformers_available

from src.trainer.config import VaeDtype, WeightDtype


def _format_gpu() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    return f"{name} (sm {major}.{minor})"


def validate_gpu_config(
    *,
    attention_mechanism: str,
    mixed_precision: WeightDtype,
    vae_dtype: VaeDtype,
) -> None:
    if attention_mechanism == "xformers" and not is_xformers_available():
        raise ValueError(
            "attention_mechanism='xformers' requires the xformers package. "
            "Install it with: uv add xformers"
        )

    if not torch.cuda.is_available():
        return

    major, _minor = torch.cuda.get_device_capability()
    gpu = _format_gpu()

    if mixed_precision == WeightDtype.BFLOAT_16 and (major < 8 or not torch.cuda.is_bf16_supported()):
        raise ValueError(
            f"mixed_precision=bfloat16 is not supported on {gpu}. "
            "bfloat16 requires Ampere or newer (sm 8.0+). Use mixed_precision: float16."
        )

    if vae_dtype == VaeDtype.BFLOAT_16 and (major < 8 or not torch.cuda.is_bf16_supported()):
        raise ValueError(
            f"vae_dtype=bfloat16 is not supported on {gpu}. "
            "Use vae_dtype: float16, float32, or auto."
        )
