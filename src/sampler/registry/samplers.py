"""Canonical sampler IDs (ComfyUI names) and per-sampler defaults."""

from __future__ import annotations

from dataclasses import dataclass

# Solver names dispatched to k-diffusion sample_* functions.
# ddim is executed as euler (ComfyUI sampler_object mapping).
SAMPLER_IDS: tuple[str, ...] = (
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "ddim",
    "heun",
    "lms",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
)

DEFAULT_SAMPLER = "euler"

# Samplers that need an extra sigma step then drop the penultimate value.
DISCARD_PENULTIMATE_SIGMA_SAMPLERS: frozenset[str] = frozenset(
    {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"}
)


@dataclass(frozen=True)
class SamplerSpec:
    id: str
    default_scheduler: str
    comfy_label: str
    reforge_label: str


SAMPLER_SPECS: tuple[SamplerSpec, ...] = (
    SamplerSpec("euler", "normal", "euler", "Euler"),
    SamplerSpec("euler_ancestral", "normal", "euler_ancestral", "Euler a"),
    SamplerSpec("dpmpp_2m", "karras", "dpmpp_2m", "DPM++ 2M"),
    SamplerSpec("ddim", "ddim_uniform", "ddim", "DDIM"),
    SamplerSpec("heun", "normal", "heun", "Heun"),
    SamplerSpec("lms", "normal", "lms", "LMS"),
    SamplerSpec("dpmpp_2s_ancestral", "karras", "dpmpp_2s_ancestral", "DPM++ 2S a"),
    SamplerSpec("dpmpp_sde", "karras", "dpmpp_sde", "DPM++ SDE"),
)

SAMPLER_SPEC_BY_ID: dict[str, SamplerSpec] = {spec.id: spec for spec in SAMPLER_SPECS}


def default_scheduler_for(sampler_id: str) -> str:
    spec = SAMPLER_SPEC_BY_ID.get(sampler_id)
    if spec is None:
        return "normal"
    return spec.default_scheduler
