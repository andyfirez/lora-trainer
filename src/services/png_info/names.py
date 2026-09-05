"""A1111/reForge infotext names for sampler + schedule type."""

from __future__ import annotations

from src.sampler.registry.legacy import LEGACY_BUNDLED_SCHEDULERS
from src.sampler.registry.profiles import (
    SamplingProfile,
    sampler_label,
    scheduler_label,
)
from src.sampler.registry.samplers import SAMPLER_SPEC_BY_ID, SAMPLER_SPECS
from src.sampler.registry.schedulers import SCHEDULER_SPEC_BY_ID, SCHEDULER_SPECS

_SAMPLER_ALIASES: dict[str, str] = {}
_SCHEDULER_ALIASES: dict[str, str] = {}
for _spec in SAMPLER_SPECS:
    _SAMPLER_ALIASES[_spec.id] = _spec.id
    _SAMPLER_ALIASES[_spec.comfy_label.lower()] = _spec.id
    _SAMPLER_ALIASES[_spec.reforge_label.lower()] = _spec.id
_SAMPLER_ALIASES["euler a"] = "euler_ancestral"
_SAMPLER_ALIASES["euler ancestral"] = "euler_ancestral"
_SAMPLER_ALIASES["dpm++ 2m"] = "dpmpp_2m"
_SAMPLER_ALIASES["dpm++"] = "dpmpp_2m"
_SAMPLER_ALIASES["dpm++ 2s a"] = "dpmpp_2s_ancestral"
_SAMPLER_ALIASES["dpm++ 2s ancestral"] = "dpmpp_2s_ancestral"
_SAMPLER_ALIASES["dpm++ sde"] = "dpmpp_sde"

for _spec in SCHEDULER_SPECS:
    _SCHEDULER_ALIASES[_spec.id] = _spec.id
    _SCHEDULER_ALIASES[_spec.comfy_label.lower()] = _spec.id
    _SCHEDULER_ALIASES[_spec.reforge_label.lower()] = _spec.id
_SCHEDULER_ALIASES["automatic"] = "automatic"
_SCHEDULER_ALIASES["schedule type"] = "automatic"

_SCHEDULER_SUFFIXES = sorted(
    {spec.reforge_label.lower() for spec in SCHEDULER_SPECS} | {spec.id for spec in SCHEDULER_SPECS},
    key=len,
    reverse=True,
)


def infotext_sampler_label(sampler_id: str, profile: SamplingProfile = SamplingProfile.REFORGE) -> str:
    spec = SAMPLER_SPEC_BY_ID.get(sampler_id)
    if spec is None:
        return sampler_id
    return sampler_label(spec, profile)


def infotext_scheduler_label(scheduler_id: str, profile: SamplingProfile = SamplingProfile.REFORGE) -> str:
    spec = SCHEDULER_SPEC_BY_ID.get(scheduler_id)
    if spec is None:
        return scheduler_id
    return scheduler_label(spec, profile)


def lookup_sampler_id(name: str) -> str | None:
    return _SAMPLER_ALIASES.get(name.strip().lower())


def lookup_scheduler_id(name: str) -> str | None:
    return _SCHEDULER_ALIASES.get(name.strip().lower())


def split_combined_sampler_name(name: str) -> tuple[str | None, str | None]:
    """Split 'DPM++ 2M Karras' into sampler + scheduler IDs."""
    text = name.strip()
    lowered = text.lower()
    for suffix in _SCHEDULER_SUFFIXES:
        token = f" {suffix}"
        if lowered.endswith(token):
            sampler = lookup_sampler_id(text[: -len(token)])
            scheduler = lookup_scheduler_id(suffix)
            if sampler is not None:
                return sampler, scheduler
    sampler = lookup_sampler_id(text)
    if sampler is not None:
        return sampler, None
    legacy = LEGACY_BUNDLED_SCHEDULERS.get(lowered)
    if legacy is not None:
        return legacy
    return None, None
