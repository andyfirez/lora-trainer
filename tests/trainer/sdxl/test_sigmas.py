"""Sigma schedule length and shape checks."""

import torch
from src.trainer.sdxl.latent_sampling.model_sampling import DiscreteEpsSampling
from src.trainer.sdxl.latent_sampling.sigmas import calculate_sigmas


def _sdxl_sampling() -> DiscreteEpsSampling:
    betas = torch.linspace(0.00085, 0.012, 1000)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return DiscreteEpsSampling.from_alphas_cumprod(alphas_cumprod)


def test_karras_and_normal_have_steps_plus_zero() -> None:
    model_sampling = _sdxl_sampling()
    karras = calculate_sigmas(model_sampling, "karras", 20)
    normal = calculate_sigmas(model_sampling, "normal", 20)
    assert len(karras) == 21
    assert float(karras[-1]) == 0.0
    assert len(normal) >= 20
    assert float(normal[-1]) == 0.0
    assert float(karras[0]) > float(karras[-2])


def test_simple_scheduler_length() -> None:
    sigmas = calculate_sigmas(_sdxl_sampling(), "simple", 8)
    assert len(sigmas) == 9
    assert float(sigmas[-1]) == 0.0
