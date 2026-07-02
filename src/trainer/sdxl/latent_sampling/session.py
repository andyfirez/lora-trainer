"""Sampling session that keeps UNet/VAE on GPU for an entire pass."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.trainer.config import TrainConfig
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
    scheduler: Any
    timesteps: Tensor
    add_time_ids: Tensor
    vae_scale_factor: int
    autocast_dtype: torch.dtype
    sample_steps: int
    sample_vae_fp32: bool
    vae_tiling_enabled: bool = False

    @classmethod
    def create(
        cls,
        *,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        scheduler: Any,
        device: torch.device,
        width: int,
        height: int,
        sample_steps: int,
        autocast_dtype: torch.dtype,
        config: TrainConfig,
        vae_scale_factor: int = 8,
        reference_add_time_ids: tuple[float, ...] | None = None,
    ) -> SDXLSamplingSession:
        vae_tiling_enabled = False
        prepare_vae_for_decode(vae, use_fp32=config.sample_vae_fp32)
        if config.sample_vae_tiling and max(width, height) >= _VAE_TILING_MIN_SIDE:
            vae.enable_tiling()
            vae_tiling_enabled = True

        unet.eval()
        scheduler.set_timesteps(sample_steps, device=device)
        if reference_add_time_ids is not None:
            add_time_ids = _add_time_ids_from_reference(reference_add_time_ids, device)
        else:
            add_time_ids = _build_add_time_ids(height, width, device)
        return cls(
            device=device,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            timesteps=scheduler.timesteps,
            add_time_ids=add_time_ids,
            vae_scale_factor=vae_scale_factor,
            autocast_dtype=autocast_dtype,
            sample_steps=sample_steps,
            sample_vae_fp32=config.sample_vae_fp32,
            vae_tiling_enabled=vae_tiling_enabled,
        )
