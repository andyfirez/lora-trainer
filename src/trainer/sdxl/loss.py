"""SDXL diffusion training loss helpers (Min-SNR weighting, noise offset)."""

import torch
from torch import Tensor


def _compute_snr(timesteps: Tensor, alphas_cumprod: Tensor) -> Tensor:
    alpha_bar = alphas_cumprod[timesteps]
    return alpha_bar / (1.0 - alpha_bar)


def min_snr_weight(
    timesteps: Tensor,
    alphas_cumprod: Tensor,
    gamma: float,
    *,
    v_prediction: bool = False,
) -> Tensor:
    """Per-sample Min-SNR loss weights; gamma <= 0 returns ones."""
    if gamma <= 0:
        return torch.ones(timesteps.shape[0], device=timesteps.device, dtype=torch.float32)

    snr = _compute_snr(timesteps, alphas_cumprod.to(device=timesteps.device))
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        return (min_snr_gamma / (snr + 1.0)).float()
    return (min_snr_gamma / snr).float()


def apply_noise_offset(latents: Tensor, noise: Tensor, offset: float) -> Tensor:
    if offset <= 0:
        return noise
    channel_noise = torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1),
        device=latents.device,
        dtype=noise.dtype,
    )
    return noise + offset * channel_noise
