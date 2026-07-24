"""Noise preparation adapted from ComfyUI 0.27.0 comfy/sample.py (GPL-3)."""

from __future__ import annotations

import torch
from torch import Tensor


def prepare_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return noise.to(device=device, dtype=dtype)
