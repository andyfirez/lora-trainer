"""Validate dtype and attention settings against the local GPU."""

import torch
from diffusers.utils.import_utils import is_xformers_available

from src.trainer.config import VaeDtype, WeightDtype
from src.trainer.sdxl.model_loader import resolve_vae_dtype

_WEIGHT_DTYPE_TO_TORCH = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}


def _format_gpu() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    return f"{name} (sm {major}.{minor})"


def _format_torch_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def _validate_reforge_sampler_dtypes(
    *,
    use_reforge_sampler: bool,
    mixed_precision: WeightDtype,
    vae_dtype: VaeDtype,
) -> None:
    if not use_reforge_sampler:
        return

    mixed_torch_dtype = _WEIGHT_DTYPE_TO_TORCH[mixed_precision]
    resolved_vae_dtype = resolve_vae_dtype(vae_dtype)
    if mixed_torch_dtype == resolved_vae_dtype:
        return

    resolved_label = (
        _format_torch_dtype(resolved_vae_dtype)
        if vae_dtype != VaeDtype.AUTO
        else f"auto→{_format_torch_dtype(resolved_vae_dtype)}"
    )
    raise ValueError(
        "use_reforge_sampler=true requires mixed_precision to match vae_dtype: "
        f"mixed_precision={mixed_precision.value}, vae_dtype={resolved_label}. "
        f"Set vae_dtype to {mixed_precision.value} or change mixed_precision."
    )


def validate_gpu_config(
    *,
    attention_mechanism: str,
    mixed_precision: WeightDtype,
    vae_dtype: VaeDtype,
    use_reforge_sampler: bool = False,
) -> None:
    if attention_mechanism == "xformers" and not is_xformers_available():
        raise ValueError(
            "attention_mechanism='xformers' requires the xformers package. "
            "Install it with: uv add xformers"
        )

    _validate_reforge_sampler_dtypes(
        use_reforge_sampler=use_reforge_sampler,
        mixed_precision=mixed_precision,
        vae_dtype=vae_dtype,
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
