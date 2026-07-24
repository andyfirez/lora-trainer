"""Build Comfy-compatible sigma schedules for SDXL sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler
from torch import Tensor

from src.trainer.sdxl.latent_sampling.comfy.constants import (
    validate_sampler_scheduler_pair,
)
from src.trainer.sdxl.latent_sampling.comfy.model_sampling import EpsModelSampling
from src.trainer.sdxl.latent_sampling.comfy.schedulers import calculate_sigmas


@dataclass(frozen=True)
class ComfySamplingPlan:
    sampler_name: str
    scheduler_name: str
    sigmas: Tensor
    model_sampling: EpsModelSampling


def build_comfy_sampling_plan(
    *,
    sampler_name: str,
    scheduler_name: str,
    steps: int,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
) -> ComfySamplingPlan:
    validate_sampler_scheduler_pair(sampler_name, scheduler_name)
    alphas_cumprod = noise_scheduler.alphas_cumprod
    if not isinstance(alphas_cumprod, torch.Tensor):
        alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
    model_sampling = EpsModelSampling(alphas_cumprod)
    sigmas = calculate_sigmas(model_sampling, scheduler_name, steps).to(device)
    return ComfySamplingPlan(
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        sigmas=sigmas,
        model_sampling=model_sampling,
    )
