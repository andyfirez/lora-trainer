from unittest.mock import MagicMock, patch

import torch

from src.trainer.sdxl.sampling import build_embed_only_sdxl_pipeline, precompute_all_sample_embeds


@patch("src.trainer.sdxl.sampling.StableDiffusionXLPipeline")
def test_build_embed_only_sdxl_pipeline_omits_text_encoders(mock_pipe_cls: MagicMock) -> None:
    vae = MagicMock()
    unet = MagicMock()
    tokenizer_1 = MagicMock()
    tokenizer_2 = MagicMock()
    scheduler = MagicMock()

    build_embed_only_sdxl_pipeline(
        vae=vae,
        unet=unet,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
    )

    mock_pipe_cls.assert_called_once_with(
        vae=vae,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )


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
    )

    assert len(results) == 2
    assert mock_encode.call_count == 3
    assert torch.equal(results[0].negative_prompt_embeds, results[1].negative_prompt_embeds)
