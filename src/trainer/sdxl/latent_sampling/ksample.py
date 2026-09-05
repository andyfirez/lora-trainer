"""Latent-space SDXL denoising (Comfy KSampler equivalent)."""

import logging
import time
from collections.abc import Callable

import torch
from torch import Tensor

from src.trainer.sdxl.latent_sampling.kdiffusion import get_sample_function
from src.trainer.sdxl.latent_sampling.model_wrapper import SDXLKDiffusionModel
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession
from src.trainer.sdxl.sampling import SamplePromptEmbeds

StepProgressCallback = Callable[[int, int], None]
LatentStepCallback = Callable[[int, int, Tensor], None]


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
    on_latent: LatentStepCallback | None = None,
    log: logging.Logger | None = None,
    log_prefix: str = "",
) -> Tensor:
    device = session.device
    prep_started_at = time.perf_counter()

    generator = torch.Generator(device=device)
    noise_seed = None if seed is None else seed + prompt_index
    if noise_seed is not None:
        generator.manual_seed(noise_seed)

    latent_channels = session.unet.config.in_channels
    noise = torch.randn(
        size=(1, latent_channels, height // session.vae_scale_factor, width // session.vae_scale_factor),
        generator=generator,
        device=device,
        dtype=session.autocast_dtype,
    )
    sigma_max = session.sigmas[0]
    latent = session.model_sampling.noise_scaling(sigma_max, noise, max_denoise=True)

    model = SDXLKDiffusionModel(
        unet=session.unet,
        model_sampling=session.model_sampling,
        embeds=embeds,
        add_time_ids=session.add_time_ids,
        guidance_scale=guidance_scale,
        autocast_dtype=session.autocast_dtype,
    )
    sample_fn = get_sample_function(session.sampler_name)
    extra_args = {"seed": noise_seed}
    total_steps = max(len(session.sigmas) - 1, 1)

    def _callback(payload: dict[str, object]) -> None:
        completed = int(payload["i"]) + 1
        current = payload["x"]
        if on_step_end is not None:
            on_step_end(completed, total_steps)
        if on_latent is not None and isinstance(current, Tensor):
            on_latent(completed, total_steps, current)

    if log is not None:
        log.info(
            "%s ksample prep (latents+embeds): %.3fs sampler=%s scheduler=%s",
            log_prefix,
            time.perf_counter() - prep_started_at,
            session.sampler_name,
            session.scheduler_name,
        )

    loop_started_at = time.perf_counter()
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    with torch.autocast(device_type=autocast_device, dtype=session.autocast_dtype):
        latent = sample_fn(
            model,
            latent,
            session.sigmas.to(dtype=latent.dtype),
            extra_args=extra_args,
            callback=_callback,
            disable=True,
        )

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
