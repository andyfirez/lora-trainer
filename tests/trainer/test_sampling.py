from unittest.mock import MagicMock, patch

import torch
from src.trainer.sdxl.sampling import precompute_all_sample_embeds


@patch("src.trainer.sdxl.sampling.encode_sdxl_prompt")
def test_precompute_all_sample_embeds_encodes_negative_once(mock_encode: MagicMock) -> None:
    mock_encode.side_effect = [
        (torch.ones(1, 2, 3), torch.ones(1, 4)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 4)),
        (torch.full((1, 2, 3), 2.0), torch.full((1, 4), 2.0)),
    ]

    results = precompute_all_sample_embeds(
        sample_prompts=["first", "second"],
        negative_prompt="bad",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        clip_skip=2,
    )

    assert len(results) == 2
    assert mock_encode.call_count == 3
    assert torch.equal(results[0].negative_prompt_embeds, results[1].negative_prompt_embeds)
