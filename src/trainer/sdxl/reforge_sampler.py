"""Minimal reForge-like preview sampler on top of diffusers components."""

from collections.abc import Callable
from typing import Literal, Optional

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from PIL import Image


_SamplerName = Literal["euler_a", "dpmpp_2m"]
_SchedulerMode = Literal["normal", "karras"]


def _build_scheduler(
    *,
    sampler_name: _SamplerName,
    scheduler_mode: _SchedulerMode,
    noise_scheduler_config: object,
) -> object:
    use_karras = scheduler_mode == "karras"
    if sampler_name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(
            noise_scheduler_config,
            use_karras_sigmas=use_karras,
        )
    if sampler_name == "dpmpp_2m":
        return DPMSolverMultistepScheduler.from_config(
            noise_scheduler_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=use_karras,
        )
    raise ValueError(f"Unsupported sample_sampler: {sampler_name}")


def _decode_latents_to_image(vae: AutoencoderKL, latents: torch.Tensor) -> Image.Image:
    decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    image = decoded[0].permute(1, 2, 0).float().cpu().numpy()
    image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image_uint8)


def generate_preview_image(
    *,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler_config: object,
    sampler_name: _SamplerName,
    scheduler_mode: _SchedulerMode,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    autocast_dtype: torch.dtype,
    device: torch.device,
    on_step_end: Optional[Callable[[int, int], None]] = None,
) -> Image.Image:
    scheduler = _build_scheduler(
        sampler_name=sampler_name,
        scheduler_mode=scheduler_mode,
        noise_scheduler_config=noise_scheduler_config,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    latent_height = height // 8
    latent_width = width // 8
    latents = torch.randn(
        (1, unet.config.in_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=autocast_dtype,
    )
    latents = latents * scheduler.init_noise_sigma

    cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    cfg_pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    cfg_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    for step_idx, timestep in enumerate(scheduler.timesteps):
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, timestep)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            noise_pred = unet(
                latent_input,
                timestep,
                encoder_hidden_states=cfg_prompt_embeds,
                added_cond_kwargs={"text_embeds": cfg_pooled_embeds, "time_ids": cfg_time_ids},
                return_dict=False,
            )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents, generator=generator, return_dict=False)[0]

        if on_step_end is not None:
            on_step_end(step_idx + 1, num_inference_steps)

    return _decode_latents_to_image(vae, latents)

