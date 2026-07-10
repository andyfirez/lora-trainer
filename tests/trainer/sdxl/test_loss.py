"""Tests for SDXL training loss helpers."""

import torch
from src.trainer.sdxl.loss import apply_noise_offset, min_snr_weight


def _make_alphas_cumprod(num_timesteps: int = 1000) -> torch.Tensor:
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def test_min_snr_weight_high_timestep() -> None:
    alphas_cumprod = _make_alphas_cumprod()
    timesteps = torch.tensor([800, 900])

    weights = min_snr_weight(timesteps, alphas_cumprod, gamma=5.0)

    assert weights.shape == (2,)
    assert torch.all(weights <= 1.0)
    assert torch.all(weights > 0.0)


def test_min_snr_gamma_zero_is_noop() -> None:
    alphas_cumprod = _make_alphas_cumprod()
    timesteps = torch.tensor([100, 500, 999])

    weights = min_snr_weight(timesteps, alphas_cumprod, gamma=0.0)

    assert torch.allclose(weights, torch.ones_like(weights))


def test_noise_offset_adds_channel_constant() -> None:
    latents = torch.zeros(2, 4, 8, 8)
    noise = torch.ones_like(latents)

    result = apply_noise_offset(latents, noise, offset=0.0357)

    assert not torch.allclose(result, noise)
    for batch_index in range(result.shape[0]):
        for channel_index in range(result.shape[1]):
            channel_slice = result[batch_index, channel_index]
            assert torch.allclose(
                channel_slice,
                torch.full_like(channel_slice, channel_slice[0, 0]),
            )


def test_noise_offset_zero_is_noop() -> None:
    latents = torch.randn(1, 4, 8, 8)
    noise = torch.randn_like(latents)

    result = apply_noise_offset(latents, noise, offset=0.0)

    assert torch.equal(result, noise)
