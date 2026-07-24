"""Supported Comfy sampler/scheduler pairs for V1."""

from __future__ import annotations

V1_SAMPLER_NAMES: frozenset[str] = frozenset({"euler", "euler_ancestral", "dpmpp_2m"})
V1_SCHEDULER_NAMES: frozenset[str] = frozenset({"simple", "karras"})

V1_SUPPORTED_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("euler", "simple"),
        ("euler_ancestral", "simple"),
        ("dpmpp_2m", "karras"),
    }
)


def validate_sampler_scheduler_pair(sampler_name: str, scheduler: str) -> None:
    if sampler_name not in V1_SAMPLER_NAMES:
        raise ValueError(
            f"Unsupported sampler_name {sampler_name!r}; supported: {sorted(V1_SAMPLER_NAMES)}"
        )
    if scheduler not in V1_SCHEDULER_NAMES:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r}; supported: {sorted(V1_SCHEDULER_NAMES)}"
        )
    pair = (sampler_name, scheduler)
    if pair not in V1_SUPPORTED_PAIRS:
        raise ValueError(
            f"Unsupported sampler/scheduler pair {sampler_name}+{scheduler}; "
            f"supported pairs: {sorted(V1_SUPPORTED_PAIRS)}"
        )
