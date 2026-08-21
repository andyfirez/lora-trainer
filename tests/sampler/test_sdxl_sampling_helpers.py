from unittest.mock import MagicMock, patch

import torch
from src.trainer.config import VaeDtype
from src.trainer.sdxl.model_loader import resolve_vae_dtype
from src.trainer.sdxl.sampling import PromptEmbedCache


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


@patch("src.trainer.sdxl.sampling.encode_sdxl_prompt")
def test_prompt_embed_cache_reuses_positive_encoding(mock_encode: MagicMock) -> None:
    mock_encode.return_value = (torch.zeros(1, 2, 3), torch.zeros(1, 4))
    cache = PromptEmbedCache()

    first = cache.get_positive(
        prompt="hello",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        clip_skip=2,
    )
    second = cache.get_positive(
        prompt="hello",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        clip_skip=2,
    )

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert mock_encode.call_count == 1


@patch("src.trainer.sdxl.sampling.encode_sdxl_prompt")
def test_prompt_embed_cache_reuses_negative_encoding(mock_encode: MagicMock) -> None:
    mock_encode.return_value = (torch.ones(1, 2, 3), torch.ones(1, 4))
    cache = PromptEmbedCache()

    first = cache.get_negative(
        negative_prompt="bad",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        clip_skip=2,
    )
    second = cache.get_negative(
        negative_prompt="bad",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        clip_skip=2,
    )

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert mock_encode.call_count == 1
