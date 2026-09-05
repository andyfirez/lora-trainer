"""Sampler/scheduler registry with Comfy and reForge UI profiles."""

from src.sampler.registry.defaults import (
    UnknownSamplerError,
    UnknownSchedulerError,
    resolve_sampler_scheduler,
)
from src.sampler.registry.legacy import (
    LEGACY_BUNDLED_SCHEDULERS,
    normalize_experiment_parameters,
)
from src.sampler.registry.profiles import (
    SamplingProfile,
    samplers_for_profile,
    schedulers_for_profile,
)
from src.sampler.registry.samplers import (
    DEFAULT_SAMPLER,
    SAMPLER_IDS,
    default_scheduler_for,
)
from src.sampler.registry.schedulers import (
    AUTOMATIC_SCHEDULER,
    DEFAULT_SCHEDULER,
    SCHEDULER_IDS,
)

__all__ = [
    "AUTOMATIC_SCHEDULER",
    "DEFAULT_SAMPLER",
    "DEFAULT_SCHEDULER",
    "LEGACY_BUNDLED_SCHEDULERS",
    "SAMPLER_IDS",
    "SCHEDULER_IDS",
    "SamplingProfile",
    "UnknownSamplerError",
    "UnknownSchedulerError",
    "default_scheduler_for",
    "normalize_experiment_parameters",
    "resolve_sampler_scheduler",
    "samplers_for_profile",
    "schedulers_for_profile",
]
