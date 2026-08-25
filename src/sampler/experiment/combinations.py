"""Build cartesian product of sweep parameter combinations."""

from __future__ import annotations

import itertools
from typing import Any

from src.sampler.experiment.lora_params import apply_lora_params
from src.sampler.experiment.models import (
    EXPERIMENT_PARAM_ORDER,
    ExperimentCombination,
    ExperimentParameters,
    LoraEntry,
    dedupe_lora_entries,
    parse_lora_entries,
    parse_lora_entry,
)
from src.sampler.lora_prompt import apply_prompt_lora_to_params


def _lora_entries_for_param(parameters: ExperimentParameters) -> list[LoraEntry]:
    param = parameters.get_param("lora_path")
    values = param.effective_values()
    if not values:
        return [LoraEntry(path=None, trigger="")]
    entries = [parse_lora_entry(value) for value in values]
    deduped = dedupe_lora_entries(entries)
    return deduped if deduped else [LoraEntry(path=None, trigger="")]


def _values_for_key(parameters: ExperimentParameters, key: str) -> list[Any]:
    if key == "lora_path":
        return _lora_entries_for_param(parameters)
    param = parameters.get_param(key)
    return param.effective_values()


def build_combinations(parameters: ExperimentParameters) -> list[ExperimentCombination]:
    """Return resolved parameter dicts for every sweep cell."""
    vary_keys = parameters.vary_keys_with_values()
    if not vary_keys:
        combo: dict[str, Any] = {}
        for key in EXPERIMENT_PARAM_ORDER:
            param = parameters.get_param(key)
            vals = param.effective_values()
            if key == "lora_path":
                entries = parse_lora_entries(vals[0]) if vals else [LoraEntry()]
                apply_lora_params(combo, entries)
            elif vals:
                combo[key] = vals[0]
            elif key == "lora_weight":
                combo.setdefault("lora_weight", 1.0)
        if not combo.get("prompt"):
            return []
        apply_prompt_lora_to_params(combo)
        return [ExperimentCombination(index=0, params=combo)]

    axes: list[tuple[str, list[Any]]] = []
    for key in EXPERIMENT_PARAM_ORDER:
        if key not in vary_keys:
            continue
        values = _values_for_key(parameters, key)
        if not values and key != "lora_path":
            continue
        axes.append((key, values))

    fixed: dict[str, Any] = {}
    for key in EXPERIMENT_PARAM_ORDER:
        if key in vary_keys:
            continue
        param = parameters.get_param(key)
        vals = param.effective_values()
        if key == "lora_path":
            if vals:
                apply_lora_params(fixed, parse_lora_entries(vals[0]))
            else:
                apply_lora_params(fixed, [LoraEntry()])
        elif vals:
            fixed[key] = vals[0]
        elif key == "lora_weight":
            fixed.setdefault("lora_weight", 1.0)

    combinations: list[ExperimentCombination] = []
    for index, combo_values in enumerate(itertools.product(*(vals for _, vals in axes))):
        params = dict(fixed)
        for (key, _), value in zip(axes, combo_values, strict=True):
            if key == "lora_path":
                apply_lora_params(params, [parse_lora_entry(value)])
            else:
                params[key] = value
        if not params.get("prompt"):
            continue
        apply_prompt_lora_to_params(params)
        combinations.append(ExperimentCombination(index=index, params=params))
    return combinations


def count_combinations(parameters: ExperimentParameters) -> int:
    return len(build_combinations(parameters))
