"""Tests for CLIP hidden-state selection."""

import torch
from src.trainer.sdxl.prompt_encoding import select_clip_hidden_state


def test_select_clip_hidden_state_clip_skip_2() -> None:
    hidden_states = (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]))
    assert torch.equal(select_clip_hidden_state(hidden_states, 2), torch.tensor([1.0]))


def test_select_clip_hidden_state_clip_skip_1() -> None:
    hidden_states = (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]))
    assert torch.equal(select_clip_hidden_state(hidden_states, 1), torch.tensor([2.0]))
