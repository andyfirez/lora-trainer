"""Latent-space SDXL denoising (Comfy KSampler equivalent)."""

import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession
from src.trainer.sdxl.sampling import SamplePromptEmbeds

StepProgressCallback = Callable[[int, int], None]


def ksample_sdxl_latent(
    session: SDXLSamplingSession,
    embeds: SamplePromptEmbeds,
    *,
    width: int,
    height: int,
    guidance_scale: float,
    seed: int | None,
    prompt_index: int,
    on_step_end: StepProgressCallback | None = None,
    log: logging.Logger | None = None,
    log_prefix: str = "",
) -> Tensor:
    device = session.device
    prep_started_at = time.perf_counter()

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed + prompt_index)

    latent_channels = session.unet.config.in_channels
    latent = torch.randn(
        size=(1, latent_channels, height // session.vae_scale_factor, width // session.vae_scale_factor),
        generator=generator,
        device=device,
        dtype=session.autocast_dtype,
    )
    latent = latent * session.scheduler.init_noise_sigma

    combined_prompt_embeds = torch.cat([embeds.negative_prompt_embeds, embeds.prompt_embeds], dim=0).to(
        dtype=session.autocast_dtype,
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat(
            [embeds.negative_pooled_prompt_embeds, embeds.pooled_prompt_embeds],
            dim=0,
        ).to(dtype=session.autocast_dtype),
        "time_ids": torch.cat([session.add_time_ids, session.add_time_ids], dim=0),
    }

    extra_step_kwargs: dict[str, Any] = {}
    if "generator" in inspect.signature(session.scheduler.step).parameters:
        extra_step_kwargs["generator"] = generator

    if log is not None:
        log.info(
            "%s ksample prep (latents+embeds): %.3fs",
            log_prefix,
            time.perf_counter() - prep_started_at,
        )

    total_steps = len(session.timesteps)
    loop_started_at = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=session.autocast_dtype):
        for step_index, timestep in enumerate(session.timesteps):
            latent_model_input = torch.cat([latent, latent], dim=0)
            latent_model_input = session.scheduler.scale_model_input(latent_model_input, timestep)

            noise_pred = session.unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=combined_prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)

            latent = session.scheduler.step(
                noise_pred,
                timestep,
                latent,
                return_dict=False,
                **extra_step_kwargs,
            )[0]

            completed = step_index + 1
            if on_step_end is not None:
                on_step_end(completed, total_steps)

    if log is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        log.info(
            "%s ksample denoise loop (GPU): %.3fs (%d steps)",
            log_prefix,
            time.perf_counter() - loop_started_at,
            total_steps,
        )

    return latent
