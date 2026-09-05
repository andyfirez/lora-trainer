"""Map the old bundled diffusers scheduler field onto sampler + schedule."""

from __future__ import annotations

from typing import Any

# Old parameters.scheduler values that selected a diffusers class, not a sigma schedule.
LEGACY_BUNDLED_SCHEDULERS: dict[str, tuple[str, str]] = {
    "euler": ("euler", "normal"),
    "euler_a": ("euler_ancestral", "normal"),
    "ddim": ("ddim", "ddim_uniform"),
    "dpm++": ("dpmpp_2m", "karras"),
}


def _as_param_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    mode = getattr(value, "mode", None)
    if mode is None:
        return None
    return {
        "mode": mode.value if hasattr(mode, "value") else mode,
        "value": getattr(value, "value", None),
        "values": list(getattr(value, "values", None) or []),
    }


def _effective_values(param: dict[str, Any] | None) -> list[Any]:
    if param is None:
        return []
    if param.get("mode") == "vary" and param.get("values"):
        return list(param["values"])
    if param.get("value") is not None:
        return [param["value"]]
    return []


def _param_with_values(template: dict[str, Any] | None, values: list[Any]) -> dict[str, Any]:
    mode = "vary" if template and template.get("mode") == "vary" and len(values) > 1 else "fixed"
    if mode == "vary":
        return {"mode": "vary", "values": values}
    return {"mode": "fixed", "value": values[0] if values else None}


def _map_legacy_value(value: Any) -> tuple[str, str] | None:
    key = str(value).strip()
    mapped = LEGACY_BUNDLED_SCHEDULERS.get(key)
    if mapped is None:
        return None
    return mapped


def normalize_experiment_parameters(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a parameters dict in place-friendly copy; keep unknown keys."""
    params = dict(data)
    sampler_param = _as_param_dict(params.get("sampler"))
    scheduler_param = _as_param_dict(params.get("scheduler"))
    sampler_explicit = "sampler" in params
    scheduler_values = _effective_values(scheduler_param)
    if not scheduler_values:
        return params

    mapped_pairs = [_map_legacy_value(value) for value in scheduler_values]
    if any(pair is None for pair in mapped_pairs):
        return params
    resolved = [pair for pair in mapped_pairs if pair is not None]
    samplers = [pair[0] for pair in resolved]
    schedules = [pair[1] for pair in resolved]
    if not sampler_explicit:
        params["sampler"] = _param_with_values(scheduler_param, samplers)
        params["scheduler"] = _param_with_values(scheduler_param, schedules)
        return params
    sampler_values = _effective_values(sampler_param)
    if not sampler_values or sampler_values == samplers:
        params["scheduler"] = _param_with_values(scheduler_param, schedules)
    return params
