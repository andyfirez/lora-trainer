"""Build cartesian product of sweep parameter combinations."""

from __future__ import annotations

import itertools
from typing import Any

from src.sampler.sweep.models import SWEEP_PARAM_ORDER, SweepCombination, SweepParameters


def _values_for_key(parameters: SweepParameters, key: str) -> list[Any]:
    param = parameters.get_param(key)
    values = param.effective_values()
    if key == "lora_path" and not values:
        return [None]
    return values


def build_combinations(parameters: SweepParameters) -> list[SweepCombination]:
    """Return resolved parameter dicts for every sweep cell."""
    vary_keys = parameters.vary_keys_with_values()
    if not vary_keys:
        combo: dict[str, Any] = {}
        for key in SWEEP_PARAM_ORDER:
            param = parameters.get_param(key)
            vals = param.effective_values()
            if vals:
                combo[key] = vals[0]
            elif key == "lora_path":
                combo[key] = None
            elif key == "lora_weight":
                combo[key] = 1.0
        if not combo.get("prompt"):
            return []
        return [SweepCombination(index=0, params=combo)]

    axes: list[tuple[str, list[Any]]] = []
    for key in SWEEP_PARAM_ORDER:
        if key not in vary_keys:
            continue
        values = _values_for_key(parameters, key)
        if not values and key != "lora_path":
            continue
        axes.append((key, values))

    fixed: dict[str, Any] = {}
    for key in SWEEP_PARAM_ORDER:
        if key in vary_keys:
            continue
        param = parameters.get_param(key)
        vals = param.effective_values()
        if vals:
            fixed[key] = vals[0]
        elif key == "lora_weight":
            fixed[key] = 1.0

    combinations: list[SweepCombination] = []
    for index, combo_values in enumerate(itertools.product(*(vals for _, vals in axes))):
        params = dict(fixed)
        for (key, _), value in zip(axes, combo_values, strict=True):
            params[key] = value
        if not params.get("prompt"):
            continue
        combinations.append(SweepCombination(index=index, params=params))
    return combinations


def count_combinations(parameters: SweepParameters) -> int:
    return len(build_combinations(parameters))
