"""ComfyUI 0.27.0-compatible SDXL sampling primitives (GPL-3 adapted)."""

from src.trainer.sdxl.latent_sampling.comfy.constants import (
    V1_SAMPLER_NAMES,
    V1_SCHEDULER_NAMES,
    V1_SUPPORTED_PAIRS,
    validate_sampler_scheduler_pair,
)
from src.trainer.sdxl.latent_sampling.comfy.model_sampling import EpsModelSampling
from src.trainer.sdxl.latent_sampling.comfy.plan import (
    ComfySamplingPlan,
    build_comfy_sampling_plan,
)
from src.trainer.sdxl.latent_sampling.comfy.runner import run_comfy_ksample

__all__ = [
    "ComfySamplingPlan",
    "EpsModelSampling",
    "V1_SAMPLER_NAMES",
    "V1_SCHEDULER_NAMES",
    "V1_SUPPORTED_PAIRS",
    "build_comfy_sampling_plan",
    "run_comfy_ksample",
    "validate_sampler_scheduler_pair",
]
