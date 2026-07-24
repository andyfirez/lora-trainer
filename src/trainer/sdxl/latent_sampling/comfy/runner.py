"""Comfy-compatible SDXL latent denoising loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from src.trainer.sdxl.latent_sampling.comfy.cfg_model import (
    DiffusersCfgContext,
    DiffusersCfgModel,
)
from src.trainer.sdxl.latent_sampling.comfy.noise import prepare_noise
from src.trainer.sdxl.latent_sampling.comfy.plan import ComfySamplingPlan
from src.trainer.sdxl.latent_sampling.comfy.samplers import run_sampler
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession
from src.trainer.sdxl.sampling import SamplePromptEmbeds

StepProgressCallback = Callable[[int, int], None]


def run_comfy_ksample(
    session: SDXLSamplingSession,
    plan: ComfySamplingPlan,
    embeds: SamplePromptEmbeds,
    *,
    width: int,
    height: int,
    guidance_scale: float,
    seed: int | None,
    on_step_end: StepProgressCallback | None = None,
    log: logging.Logger | None = None,
    log_prefix: str = "",
) -> Tensor:
    device = session.device
    prep_started_at = time.perf_counter()

    if seed is None:
        seed = 0

    latent_channels = session.unet.config.in_channels
    latent_shape = (
        1,
        latent_channels,
        height // session.vae_scale_factor,
        width // session.vae_scale_factor,
    )
    noise = prepare_noise(
        latent_shape,
        seed=seed,
        dtype=session.autocast_dtype,
        device=device,
    )
    empty_latent = torch.zeros(latent_shape, device=device, dtype=session.autocast_dtype)
    sigmas = plan.sigmas
    max_denoise = plan.model_sampling.max_denoise(sigmas[0])
    x = plan.model_sampling.noise_scaling(
        sigmas[0],
        noise,
        empty_latent,
        max_denoise=max_denoise,
    )

    cfg_model = DiffusersCfgModel(
        DiffusersCfgContext(
            unet=session.unet,
            model_sampling=plan.model_sampling,
            embeds=embeds,
            guidance_scale=guidance_scale,
            add_time_ids=session.add_time_ids,
            autocast_dtype=session.autocast_dtype,
            device=device,
        )
    )

    total_steps = len(sigmas) - 1

    def callback(payload: dict[str, Any]) -> None:
        completed = int(payload["i"]) + 1
        if on_step_end is not None:
            on_step_end(completed, total_steps)

    if log is not None:
        log.info(
            "%s comfy ksample prep (latents+embeds): %.3fs",
            log_prefix,
            time.perf_counter() - prep_started_at,
        )

    loop_started_at = time.perf_counter()
    with torch.no_grad():
        samples = run_sampler(
            plan.sampler_name,
            cfg_model,
            x,
            sigmas,
            extra_args={"seed": seed},
            callback=callback,
        )
    samples = plan.model_sampling.inverse_noise_scaling(sigmas[-1], samples)

    if log is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        log.info(
            "%s comfy ksample denoise loop (GPU): %.3fs (%d steps, %s+%s)",
            log_prefix,
            time.perf_counter() - loop_started_at,
            total_steps,
            plan.sampler_name,
            plan.scheduler_name,
        )

    return samples
