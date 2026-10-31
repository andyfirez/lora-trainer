"""Flow-matching helpers shared by Flux.1 and FLUX.2 Klein: timestep sampling and latent packing."""

from __future__ import annotations

import torch
from torch import Tensor

from src.trainer.config import FluxTimestepDistribution


def shift_sigmas(sigmas: Tensor, shift: float) -> Tensor:
    """Resolution-independent timestep shift: ``s*t / (1 + (s-1)*t)``."""
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def sample_sigmas(
    batch_size: int,
    distribution: FluxTimestepDistribution,
    shift: float,
    device: torch.device,
) -> Tensor:
    """Sample noise levels in (0, 1); 1.0 is pure noise."""
    if distribution == FluxTimestepDistribution.LOGIT_NORMAL:
        base = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        base = torch.rand(batch_size, device=device)
    return shift_sigmas(base, shift).clamp(1e-4, 1.0)


def add_flow_noise(latents: Tensor, noise: Tensor, sigmas: Tensor) -> Tensor:
    sigma = sigmas.view(-1, *([1] * (latents.ndim - 1))).to(latents.dtype)
    return (1.0 - sigma) * latents + sigma * noise


# ------------------------------------------------------------------------------ Flux.1


def flux1_pack(latents: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, (H/2)*(W/2), C*4]"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def flux1_unpack(packed: Tensor, height: int, width: int) -> Tensor:
    """Inverse of :func:`flux1_pack` for latent height/width (before 2x2 packing)."""
    b, _, channels = packed.shape
    latents = packed.view(b, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, channels // 4, height, width)


def flux1_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    h2, w2 = height // 2, width // 2
    ids = torch.zeros(h2, w2, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(h2, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(w2, device=device, dtype=dtype)[None, :]
    return ids.reshape(h2 * w2, 3)


# ------------------------------------------------------------------------- FLUX.2 Klein


def flux2_patchify(latents: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, C*4, H/2, W/2]"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def flux2_unpatchify(latents: Tensor) -> Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def flux2_pack(latents: Tensor) -> Tensor:
    """[B, C, H, W] (already patchified) -> [B, H*W, C]"""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def flux2_unpack(packed: Tensor, height: int, width: int) -> Tensor:
    b, _, c = packed.shape
    return packed.reshape(b, height, width, c).permute(0, 3, 1, 2)


def flux2_image_ids(batch: int, height: int, width: int, device: torch.device) -> Tensor:
    t = torch.arange(1, device=device)
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    l_ = torch.arange(1, device=device)
    ids = torch.cartesian_prod(t, h, w, l_)
    return ids.unsqueeze(0).expand(batch, -1, -1)


def flux2_text_ids(batch: int, length: int, device: torch.device) -> Tensor:
    t = torch.arange(1, device=device)
    h = torch.arange(1, device=device)
    w = torch.arange(1, device=device)
    l_ = torch.arange(length, device=device)
    ids = torch.cartesian_prod(t, h, w, l_)
    return ids.unsqueeze(0).expand(batch, -1, -1)
