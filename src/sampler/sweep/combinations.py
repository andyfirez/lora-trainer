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
    lora_entry_to_param_value,
    parse_lora_entries,
    parse_lora_entry,
    parse_trigger_words,
)


def _lora_entries_for_param(parameters: SweepParameters) -> list[LoraEntry]:
    param = parameters.get_param("lora_path")
    values = param.effective_values()
    if not values:
        return [LoraEntry(path=None, trigger="")]
    entries = [parse_lora_entry(value) for value in values]
    deduped = dedupe_lora_entries(entries)
    return deduped if deduped else [LoraEntry(path=None, trigger="")]


def _stack_trigger(entries: list[LoraEntry]) -> str:
    words: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for word in parse_trigger_words(entry.trigger):
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(word)
    return ", ".join(words)


def _apply_lora_params(params: dict[str, Any], entries: list[LoraEntry]) -> None:
    file_entries = [entry for entry in entries if lora_entry_path(entry)]
    if len(file_entries) > 1:
        params["lora_path"] = lora_entry_path(file_entries[0])
        params["lora_stack"] = [lora_entry_to_param_value(entry) for entry in file_entries]
        params["lora_trigger"] = _stack_trigger(file_entries)
        return
    entry = file_entries[0] if file_entries else (entries[0] if entries else LoraEntry())
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
                entries = parse_lora_entries(vals[0]) if vals else [LoraEntry()]
                _apply_lora_params(combo, entries)
            elif vals:
                combo[key] = vals[0]
            elif key == "lora_weight":
                combo.setdefault("lora_weight", 1.0)
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
                _apply_lora_params(fixed, parse_lora_entries(vals[0]))
            else:
                _apply_lora_params(fixed, [LoraEntry()])
        elif vals:
            fixed[key] = vals[0]
        elif key == "lora_weight":
            fixed.setdefault("lora_weight", 1.0)

    combinations: list[SweepCombination] = []
    for index, combo_values in enumerate(itertools.product(*(vals for _, vals in axes))):
        params = dict(fixed)
        for (key, _), value in zip(axes, combo_values, strict=True):
            if key == "lora_path":
                _apply_lora_params(params, [parse_lora_entry(value)])
            else:
                params[key] = value
        if not params.get("prompt"):
            continue
        combinations.append(SweepCombination(index=index, params=params))
    return combinations


def count_combinations(parameters: SweepParameters) -> int:
    return len(build_combinations(parameters))
