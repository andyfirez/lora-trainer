from unittest.mock import MagicMock

import torch

from src.sampler.sdxl.service import SDXLLoRASampler
from src.trainer.config import TrainConfig


def test_get_cached_prompt_embeds_reuses_encoding() -> None:
    config = TrainConfig(sample_prompts=["a"], use_reforge_sampler=True)
    sampler = SDXLLoRASampler(
        config,
        lora_paths=[],
        output_dir=MagicMock(),
    )
    encode_prompt = MagicMock(
        side_effect=[
            (torch.zeros(1, 2, 3), torch.zeros(1, 4)),
            (torch.ones(1, 2, 3), torch.ones(1, 4)),
        ]
    )
    sampler._encode_prompt = encode_prompt

    first = sampler._get_cached_prompt_embeds(
        prompt="hello",
        negative_prompt="bad",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        autocast_dtype=torch.float32,
    )
    second = sampler._get_cached_prompt_embeds(
        prompt="hello",
        negative_prompt="bad",
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=MagicMock(),
        device=torch.device("cpu"),
        autocast_dtype=torch.float32,
    )

    assert first is second
    assert encode_prompt.call_count == 2
