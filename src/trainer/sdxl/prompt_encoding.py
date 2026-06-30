"""Shared CLIP hidden-state selection for SDXL text encoding."""

from torch import Tensor


def select_clip_hidden_state(hidden_states: tuple[Tensor, ...], clip_skip: int) -> Tensor:
    return hidden_states[-clip_skip]
