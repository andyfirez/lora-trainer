"""Canonical scheduler IDs (ComfyUI sigma-schedule names)."""

from __future__ import annotations

from dataclasses import dataclass

SCHEDULER_IDS: tuple[str, ...] = (
    "normal",
    "karras",
    "exponential",
    "simple",
    "sgm_uniform",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
)

AUTOMATIC_SCHEDULER = "automatic"
DEFAULT_SCHEDULER = "normal"

# Extra reForge-only IDs that share Comfy execution when implemented.
REFORGE_EXTRA_SCHEDULER_IDS: tuple[str, ...] = (AUTOMATIC_SCHEDULER,)


@dataclass(frozen=True)
class SchedulerSpec:
    id: str
    comfy_label: str
    reforge_label: str


SCHEDULER_SPECS: tuple[SchedulerSpec, ...] = (
    SchedulerSpec("normal", "normal", "Normal"),
    SchedulerSpec("karras", "karras", "Karras"),
    SchedulerSpec("exponential", "exponential", "Exponential"),
    SchedulerSpec("simple", "simple", "Simple"),
    SchedulerSpec("sgm_uniform", "sgm_uniform", "SGM Uniform"),
    SchedulerSpec("ddim_uniform", "ddim_uniform", "DDIM"),
    SchedulerSpec("beta", "beta", "Beta"),
    SchedulerSpec("linear_quadratic", "linear_quadratic", "Linear Quadratic"),
    SchedulerSpec("kl_optimal", "kl_optimal", "KL Optimal"),
)

SCHEDULER_SPEC_BY_ID: dict[str, SchedulerSpec] = {spec.id: spec for spec in SCHEDULER_SPECS}

AUTOMATIC_SPEC = SchedulerSpec(AUTOMATIC_SCHEDULER, "automatic", "Automatic")
