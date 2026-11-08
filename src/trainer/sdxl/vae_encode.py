"""VAE encode for SDXL training (dtype handling matches vae_decode)."""

import torch
from diffusers import AutoencoderKL
from torch import Tensor


def encode_pixel_values_to_latents(
    vae: AutoencoderKL,
    pixel_values: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    """Encode pixels to scaled latents; input dtype matches VAE weights."""
    pixel_values = pixel_values.to(device)
    if vae.dtype == torch.float32:
        latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
    else:
        with torch.autocast(device_type=device.type, dtype=vae.dtype):
            latents = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
    return latents * vae.config.scaling_factor
