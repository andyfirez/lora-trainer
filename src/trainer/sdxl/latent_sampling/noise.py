"""Initial-noise generation matching each real app's RNG defaults.

ComfyUI (``comfy/sample.py::prepare_noise``) always seeds a CPU
``torch.Generator`` and samples float32 noise on CPU before moving/casting it
to the target device/dtype. reForge, with its default ``randn_source="GPU"``
setting (``modules/rng.py::create_generator``), seeds a generator directly on
the target device and samples there in the target dtype. Everything else in
the denoising loop (sigma schedules, CFG combine, solver math) is already
shared between both profiles — this is the one place their defaults diverge
enough to matter for reproducing a given app's output for a given seed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.sampler.registry.profiles import SamplingProfile


def create_initial_noise(
    profile: SamplingProfile,
    seed: int | None,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Sample initial latent noise, matching the given profile's RNG behavior.

    ``COMFY``: CPU-seeded generator, sampled on CPU, then moved to ``device``
    (matches real ComfyUI's ``prepare_noise``).

    ``REFORGE``: generator seeded directly on ``device`` (matches real
    reForge's default ``randn_source="GPU"`` RNG source).
    """
    if profile == SamplingProfile.COMFY:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        noise = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
        return noise.to(device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)
