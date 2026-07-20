"""Build cartesian product of sweep parameter combinations."""

from __future__ import annotations

import itertools
from typing import Any

from src.sampler.sweep.models import (
    SWEEP_PARAM_ORDER,
    LoraEntry,
    SweepCombination,
    SweepParameters,
    dedupe_lora_entries,
    lora_entry_path,
    parse_lora_entry,
)


def _lora_entries_for_param(parameters: SweepParameters) -> list[LoraEntry]:
    param = parameters.get_param("lora_path")
    values = param.effective_values()
    if not values:
        return [LoraEntry(path=None, trigger="")]
    entries = [parse_lora_entry(value) for value in values]
    deduped = dedupe_lora_entries(entries)
    return deduped if deduped else [LoraEntry(path=None, trigger="")]


def _apply_lora_params(params: dict[str, Any], entry: LoraEntry) -> None:
    params["lora_path"] = lora_entry_path(entry)
    params["lora_trigger"] = entry.trigger


def _values_for_key(parameters: SweepParameters, key: str) -> list[Any]:
    if key == "lora_path":
        return _lora_entries_for_param(parameters)
    param = parameters.get_param(key)
    return param.effective_values()


def build_combinations(parameters: SweepParameters) -> list[SweepCombination]:
    """Return resolved parameter dicts for every sweep cell."""
    vary_keys = parameters.vary_keys_with_values()
    if not vary_keys:
        combo: dict[str, Any] = {}
        for key in SWEEP_PARAM_ORDER:
            param = parameters.get_param(key)
            vals = param.effective_values()
            if key == "lora_path":
                entry = parse_lora_entry(vals[0]) if vals else LoraEntry(path=None, trigger="")
                _apply_lora_params(combo, entry)
            elif vals:
                combo[key] = vals[0]
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
        if key == "lora_path":
            if vals:
                _apply_lora_params(fixed, parse_lora_entry(vals[0]))
            else:
                _apply_lora_params(fixed, LoraEntry(path=None, trigger=""))
        elif vals:
            fixed[key] = vals[0]
        elif key == "lora_weight":
            fixed[key] = 1.0

    combinations: list[SweepCombination] = []
    for index, combo_values in enumerate(itertools.product(*(vals for _, vals in axes))):
        params = dict(fixed)
        for (key, _), value in zip(axes, combo_values, strict=True):
            if key == "lora_path":
                _apply_lora_params(params, parse_lora_entry(value))
            else:
                params[key] = value
        if not params.get("prompt"):
            continue
        combinations.append(SweepCombination(index=index, params=params))
    return combinations


def count_combinations(parameters: SweepParameters) -> int:
    return len(build_combinations(parameters))
