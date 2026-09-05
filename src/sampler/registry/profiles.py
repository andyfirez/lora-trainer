"""Sampling UI profiles: labels and extra options, not execution."""

from __future__ import annotations

from enum import StrEnum

from src.sampler.registry.samplers import SAMPLER_SPECS, SamplerSpec
from src.sampler.registry.schedulers import (
    AUTOMATIC_SPEC,
    SCHEDULER_SPECS,
    SchedulerSpec,
)


class SamplingProfile(StrEnum):
    COMFY = "comfy"
    REFORGE = "reforge"


def sampler_label(spec: SamplerSpec, profile: SamplingProfile) -> str:
    if profile == SamplingProfile.REFORGE:
        return spec.reforge_label
    return spec.comfy_label


def scheduler_label(spec: SchedulerSpec, profile: SamplingProfile) -> str:
    if profile == SamplingProfile.REFORGE:
        return spec.reforge_label
    return spec.comfy_label


def samplers_for_profile(profile: SamplingProfile) -> tuple[SamplerSpec, ...]:
    _ = profile
    return SAMPLER_SPECS


def schedulers_for_profile(profile: SamplingProfile) -> tuple[SchedulerSpec, ...]:
    if profile == SamplingProfile.REFORGE:
        return (AUTOMATIC_SPEC, *SCHEDULER_SPECS)
    return SCHEDULER_SPECS
