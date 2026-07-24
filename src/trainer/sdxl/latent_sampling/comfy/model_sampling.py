"""SDXL EPS model sampling math adapted from ComfyUI 0.27.0 comfy/model_sampling.py (GPL-3)."""

from __future__ import annotations

import torch
from torch import Tensor


def _reshape_sigma(sigma: Tensor, ndim: int) -> Tensor:
    if sigma.numel() == 1:
        return sigma.reshape(())
    return sigma.reshape(sigma.shape[:1] + (1,) * (ndim - 1))


class EpsModelSampling:
    """Discrete EPS sigma schedule derived from DDPM alphas_cumprod."""

    sigma_data: float = 1.0

    def __init__(self, alphas_cumprod: Tensor) -> None:
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.sigmas = sigmas.float()
        self.log_sigmas = self.sigmas.log()
        self.num_timesteps = int(self.sigmas.shape[0])

    @property
    def sigma_min(self) -> float:
        return float(self.sigmas[-1])

    @property
    def sigma_max(self) -> float:
        return float(self.sigmas[0])

    def timestep(self, sigma: Tensor) -> Tensor:
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def diffusers_timestep(self, sigma: Tensor) -> Tensor:
        index = self.timestep(sigma)
        return (self.num_timesteps - 1 - index).to(dtype=torch.long, device=sigma.device)

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        sigma = _reshape_sigma(sigma, noise.ndim)
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma: Tensor, model_output: Tensor, model_input: Tensor) -> Tensor:
        sigma = _reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(
        self,
        sigma: Tensor,
        noise: Tensor,
        latent_image: Tensor,
        *,
        max_denoise: bool = False,
    ) -> Tensor:
        sigma = _reshape_sigma(sigma, noise.ndim)
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma**2.0)
        else:
            noise = noise * sigma
        return noise + latent_image

    def inverse_noise_scaling(self, sigma: Tensor, latent: Tensor) -> Tensor:
        return latent

    def max_denoise(self, sigma: Tensor) -> bool:
        sigma_value = float(sigma.reshape(-1)[0])
        return sigma_value >= self.sigma_max or abs(sigma_value - self.sigma_max) < 1e-5 * max(self.sigma_max, 1.0)
