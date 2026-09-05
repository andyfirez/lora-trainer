"""Sampling session that keeps UNet/VAE on GPU for an entire pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.sampler.registry import resolve_sampler_scheduler
from src.sampler.registry.profiles import SamplingProfile
from src.trainer.inference_config import SDXLInferenceConfig
from src.trainer.sdxl.latent_sampling.model_sampling import DiscreteEpsSampling
from src.trainer.sdxl.latent_sampling.sigmas import calculate_sigmas
from src.trainer.sdxl.sampling import prepare_vae_for_decode

_VAE_TILING_MIN_SIDE = 768


def _build_add_time_ids(height: int, width: int, device: torch.device) -> Tensor:
    return torch.tensor(
        [
            height,
            width,
            0,
            0,
            height,
            width,
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)


def _add_time_ids_from_reference(
    reference_add_time_ids: tuple[float, ...],
    device: torch.device,
) -> Tensor:
    if len(reference_add_time_ids) != 6:
        raise ValueError(f"reference_add_time_ids must have 6 elements, got {len(reference_add_time_ids)}")
    return torch.tensor(list(reference_add_time_ids), device=device, dtype=torch.float32).unsqueeze(0)


@dataclass
class SDXLSamplingSession:
    device: torch.device
    unet: torch.nn.Module
    vae: torch.nn.Module
    model_sampling: DiscreteEpsSampling
    sigmas: Tensor
    sampler_name: str
    scheduler_name: str
    add_time_ids: Tensor
    vae_scale_factor: int
    autocast_dtype: torch.dtype
    sample_steps: int
    sample_vae_fp32: bool
    generation_profile: SamplingProfile = SamplingProfile.COMFY
    vae_tiling_enabled: bool = False

    @classmethod
    def create(
        cls,
        *,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        noise_scheduler: Any,
        device: torch.device,
        width: int,
        height: int,
        sample_steps: int,
        autocast_dtype: torch.dtype,
        config: SDXLInferenceConfig,
        vae_scale_factor: int = 8,
        reference_add_time_ids: tuple[float, ...] | None = None,
    ) -> SDXLSamplingSession:
        vae_tiling_enabled = False
        prepare_vae_for_decode(vae, use_fp32=config.sample_vae_fp32)
        if config.sample_vae_tiling and max(width, height) >= _VAE_TILING_MIN_SIDE:
            vae.enable_tiling()
            vae_tiling_enabled = True

        unet.eval()
        sampler_name, scheduler_name = resolve_sampler_scheduler(
            config.sample_sampler,
            config.sample_scheduler,
        )
        model_sampling = DiscreteEpsSampling.from_noise_scheduler(noise_scheduler).to(device)
        sigmas = calculate_sigmas(
            model_sampling,
            scheduler_name,
            sample_steps,
            sampler_name=sampler_name,
        ).to(device=device, dtype=torch.float32)
        if reference_add_time_ids is not None:
            add_time_ids = _add_time_ids_from_reference(reference_add_time_ids, device)
        else:
            add_time_ids = _build_add_time_ids(height, width, device)
        return cls(
            device=device,
            unet=unet,
            vae=vae,
            model_sampling=model_sampling,
            sigmas=sigmas,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            add_time_ids=add_time_ids,
            vae_scale_factor=vae_scale_factor,
            autocast_dtype=autocast_dtype,
            sample_steps=sample_steps,
            sample_vae_fp32=config.sample_vae_fp32,
            generation_profile=config.generation_profile,
            vae_tiling_enabled=vae_tiling_enabled,
        )
