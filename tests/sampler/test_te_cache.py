from unittest.mock import MagicMock, patch

import torch
from src.trainer.sdxl.sampling import PromptEmbedCache


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
