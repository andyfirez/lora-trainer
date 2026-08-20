from unittest.mock import patch

import pytest
from src.trainer.config import VaeDtype, WeightDtype
from src.gpu.validation import validate_gpu_config


@patch("src.gpu.validation.is_xformers_available", return_value=False)
def test_validate_gpu_config_xformers_missing_package(_xformers: object) -> None:
    with pytest.raises(ValueError, match="requires the xformers package"):
        validate_gpu_config(
            attention_mechanism="xformers",
            mixed_precision=WeightDtype.FLOAT_16,
            vae_dtype=VaeDtype.AUTO,
        )


@patch("src.gpu.validation.torch.cuda.is_available", return_value=True)
@patch("src.gpu.validation.torch.cuda.get_device_capability", return_value=(7, 5))
@patch("src.gpu.validation.torch.cuda.get_device_name", return_value="RTX 2070")
@patch("src.gpu.validation.torch.cuda.is_bf16_supported", return_value=False)
@patch("src.gpu.validation.is_xformers_available", return_value=True)
def test_validate_gpu_config_bf16_rejected_on_turing(
    _xformers: object,
    _bf16: object,
    _name: object,
    _capability: object,
    _cuda: object,
) -> None:
    with pytest.raises(ValueError, match="mixed_precision=bfloat16 is not supported"):
        validate_gpu_config(
            attention_mechanism="sdpa",
            mixed_precision=WeightDtype.BFLOAT_16,
            vae_dtype=VaeDtype.AUTO,
        )


@patch("src.gpu.validation.torch.cuda.is_available", return_value=True)
@patch("src.gpu.validation.torch.cuda.get_device_capability", return_value=(7, 5))
@patch("src.gpu.validation.torch.cuda.get_device_name", return_value="RTX 2070")
@patch("src.gpu.validation.torch.cuda.is_bf16_supported", return_value=False)
@patch("src.gpu.validation.is_xformers_available", return_value=True)
def test_validate_gpu_config_float16_xformers_ok_on_turing(
    _xformers: object,
    _bf16: object,
    _name: object,
    _capability: object,
    _cuda: object,
) -> None:
    validate_gpu_config(
        attention_mechanism="xformers",
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.AUTO,
    )


@patch("src.gpu.validation.torch.cuda.is_available", return_value=False)
@patch("src.gpu.validation.is_xformers_available", return_value=True)
def test_validate_gpu_config_skips_gpu_checks_without_cuda(
    _xformers: object,
    _cuda: object,
) -> None:
    validate_gpu_config(
        attention_mechanism="xformers",
        mixed_precision=WeightDtype.BFLOAT_16,
        vae_dtype=VaeDtype.BFLOAT_16,
    )
