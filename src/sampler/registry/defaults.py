"""Resolve automatic-mode scheduler defaults (reForge-style)."""

from __future__ import annotations

from src.sampler.registry.samplers import (
    DEFAULT_SAMPLER,
    SAMPLER_SPEC_BY_ID,
    default_scheduler_for,
)
from src.sampler.registry.schedulers import (
    AUTOMATIC_SCHEDULER,
    DEFAULT_SCHEDULER,
    SCHEDULER_SPEC_BY_ID,
)


class UnknownSamplerError(ValueError):
    pass


class UnknownSchedulerError(ValueError):
    pass


def resolve_sampler_scheduler(
    sampler_id: str,
    scheduler_id: str,
    *,
    profile: str | None = None,
) -> tuple[str, str]:
    """Return canonical (sampler, scheduler) IDs, expanding Automatic."""
    sampler = (sampler_id or DEFAULT_SAMPLER).strip()
    scheduler = (scheduler_id or DEFAULT_SCHEDULER).strip()
    if sampler not in SAMPLER_SPEC_BY_ID:
        raise UnknownSamplerError(f"Unknown sampler {sampler!r}")
    if scheduler == AUTOMATIC_SCHEDULER:
        scheduler = default_scheduler_for(sampler)
    if scheduler not in SCHEDULER_SPEC_BY_ID:
        message = f"Unknown scheduler {scheduler!r}"
        if profile:
            message = f"{message} (not supported in {profile} profile)"
        raise UnknownSchedulerError(message)
    return sampler, scheduler
