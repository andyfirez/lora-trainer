"""Latent-space SDXL denoising via Comfy-compatible k-diffusion backend."""

import logging
import time
from collections.abc import Callable

from torch import Tensor

from src.trainer.sdxl.latent_sampling.comfy.runner import run_comfy_ksample
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
    del prompt_index
    return run_comfy_ksample(
        session,
        session.sampling_plan,
        embeds,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        seed=seed,
        on_step_end=on_step_end,
        log=log,
        log_prefix=log_prefix,
    )
