"""Sigma schedules adapted from ComfyUI 0.27.0 comfy/samplers.py and k_diffusion (GPL-3)."""

from __future__ import annotations

import torch
from torch import Tensor

from src.trainer.sdxl.latent_sampling.comfy.model_sampling import EpsModelSampling


def append_zero(sigmas: Tensor) -> Tensor:
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def simple_scheduler(model_sampling: EpsModelSampling, steps: int) -> Tensor:
    sigs: list[float] = []
    step_size = len(model_sampling.sigmas) / steps
    for index in range(steps):
        sigs.append(float(model_sampling.sigmas[-(1 + int(index * step_size))]))
    sigs.append(0.0)
    return torch.tensor(sigs, dtype=torch.float32)


def get_sigmas_karras(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def calculate_sigmas(model_sampling: EpsModelSampling, scheduler_name: str, steps: int) -> Tensor:
    if scheduler_name == "simple":
        return simple_scheduler(model_sampling, steps)
    if scheduler_name == "karras":
        return get_sigmas_karras(
            steps,
            model_sampling.sigma_min,
            model_sampling.sigma_max,
            device="cpu",
        )
    raise ValueError(f"Unsupported scheduler {scheduler_name!r}")
