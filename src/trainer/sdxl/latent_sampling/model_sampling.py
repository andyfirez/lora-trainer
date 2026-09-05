"""Discrete EPS sigma tables from a DDPM alphas_cumprod (Comfy ModelSamplingDiscrete)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DiscreteEpsSampling:
    sigmas: Tensor
    log_sigmas: Tensor
    sigma_data: float = 1.0

    @classmethod
    def from_alphas_cumprod(cls, alphas_cumprod: Tensor) -> DiscreteEpsSampling:
        alphas = alphas_cumprod.float().clamp(min=1e-8, max=1.0 - 1e-8)
        sigmas = ((1.0 - alphas) / alphas).sqrt()
        return cls(sigmas=sigmas, log_sigmas=sigmas.log())

    @classmethod
    def from_noise_scheduler(cls, noise_scheduler: object) -> DiscreteEpsSampling:
        alphas_cumprod = getattr(noise_scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            raise ValueError("noise_scheduler is missing alphas_cumprod")
        if not isinstance(alphas_cumprod, Tensor):
            alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        return cls.from_alphas_cumprod(alphas_cumprod)

    @property
    def sigma_min(self) -> Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> Tensor:
        return self.sigmas[-1]

    def to(self, device: torch.device) -> DiscreteEpsSampling:
        return DiscreteEpsSampling(
            sigmas=self.sigmas.to(device=device),
            log_sigmas=self.log_sigmas.to(device=device),
            sigma_data=self.sigma_data,
        )

    def timestep(self, sigma: Tensor) -> Tensor:
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep: Tensor) -> Tensor:
        t = torch.clamp(
            timestep.float().to(self.log_sigmas.device),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        weight = t.frac()
        log_sigma = (1 - weight) * self.log_sigmas[low_idx] + weight * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        sigma = _reshape_sigma(sigma, noise.ndim)
        return noise / (sigma**2 + self.sigma_data**2).sqrt()

    def calculate_denoised(self, sigma: Tensor, model_output: Tensor, model_input: Tensor) -> Tensor:
        sigma = _reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma: Tensor, noise: Tensor, *, max_denoise: bool = True) -> Tensor:
        sigma = _reshape_sigma(sigma, noise.ndim)
        if max_denoise:
            return noise * (1.0 + sigma**2).sqrt()
        return noise * sigma


def _reshape_sigma(sigma: Tensor, noise_dim: int) -> Tensor:
    if sigma.nelement() == 1:
        return sigma.view(())
    return sigma.view(sigma.shape[:1] + (1,) * (noise_dim - 1))
