"""Tests for TE disk cache clip_skip invalidation."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from src.trainer.sdxl.te_cache import build_te_cache


def test_te_disk_cache_reencodes_when_clip_skip_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"png")
    npz_path = tmp_path / "img_te.npz"
    np.savez(
        npz_path,
        prompt_embeds=np.zeros((1, 2, 2048), dtype=np.float32),
        pooled_prompt_embeds=np.zeros((1, 1280), dtype=np.float32),
        clip_skip=np.array(1),
    )

    text_encoder_1 = MagicMock()
    text_encoder_2 = MagicMock()
    enc1_out = MagicMock()
    enc1_out.hidden_states = (torch.zeros(1), torch.ones(1))
    enc2_out = MagicMock()
    enc2_out.hidden_states = (torch.zeros(1), torch.full((1,), 2.0))
    enc2_out.__getitem__ = MagicMock(return_value=torch.full((1,), 3.0))
    text_encoder_1.return_value = enc1_out
    text_encoder_2.return_value = enc2_out

    tokenizer_1 = MagicMock()
    tokenizer_1.model_max_length = 77
    tokenizer_1.return_value = MagicMock(input_ids=torch.zeros(1, 77, dtype=torch.long))
    tokenizer_2 = MagicMock()
    tokenizer_2.model_max_length = 77
    tokenizer_2.return_value = MagicMock(input_ids=torch.zeros(1, 77, dtype=torch.long))

    cache = build_te_cache(
        [(image_path, "caption")],
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        torch.device("cpu"),
        torch.float32,
        clip_skip=2,
        to_disk=True,
    )

    assert "caption" in cache
    text_encoder_1.assert_called_once()
    text_encoder_2.assert_called_once()
    reloaded = np.load(npz_path)
    assert int(reloaded["clip_skip"]) == 2
