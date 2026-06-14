from unittest.mock import patch

import torch

from src.trainer.config import VaeDtype
from src.trainer.sdxl.model_loader import resolve_vae_dtype


@patch("src.trainer.sdxl.model_loader.torch.cuda.is_available", return_value=True)
@patch("src.trainer.sdxl.model_loader.torch.cuda.get_device_capability", return_value=(8, 6))
@patch("src.trainer.sdxl.model_loader.torch.cuda.is_bf16_supported", return_value=True)
def test_resolve_vae_dtype_auto_uses_bf16_on_ampere(
    _bf16_supported: object,
    _device_capability: object,
    _cuda_available: object,
) -> None:
    assert resolve_vae_dtype(VaeDtype.AUTO) == torch.bfloat16


@patch("src.trainer.sdxl.model_loader.torch.cuda.is_available", return_value=False)
def test_resolve_vae_dtype_auto_falls_back_to_fp32_without_cuda(_cuda_available: object) -> None:
    assert resolve_vae_dtype(VaeDtype.AUTO) == torch.float32


def test_resolve_vae_dtype_explicit_values() -> None:
    assert resolve_vae_dtype(VaeDtype.FLOAT_16) == torch.float16
    assert resolve_vae_dtype(VaeDtype.FLOAT_32) == torch.float32
