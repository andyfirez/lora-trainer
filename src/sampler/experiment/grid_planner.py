"""Plan 2D grid layouts from sweep combinations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.sampler.sweep.combinations import build_combinations
from src.sampler.sweep.models import SWEEP_PARAM_ORDER, GridLayout, SweepCombination, SweepParameters, lora_axis_display


@dataclass
class GridPlan:
    index: int
    slice_params: dict[str, Any] = field(default_factory=dict)
    x_axis: str = "prompt"
    y_axis: str = "lora_weight"
    x_values: list[Any] = field(default_factory=list)
    y_values: list[Any] = field(default_factory=list)
    cells: list[list[int | None]] = field(default_factory=list)
    title: str = ""


def _resolve_axes(
    parameters: SweepParameters,
    grid: GridLayout,
    vary_keys: list[str],
) -> tuple[str, str, list[str]]:
    if len(vary_keys) == 0:
        return "prompt", "prompt", []
    if len(vary_keys) == 1:
        return vary_keys[0], vary_keys[0], vary_keys
    x_axis = grid.x_axis or vary_keys[0]
    y_axis = grid.y_axis or (vary_keys[1] if len(vary_keys) > 1 else vary_keys[0])
    if x_axis not in vary_keys:
        x_axis = vary_keys[0]
    if y_axis not in vary_keys:
        y_axis = vary_keys[1] if len(vary_keys) > 1 else vary_keys[0]
    if x_axis == y_axis and len(vary_keys) > 1:
        y_axis = next(k for k in vary_keys if k != x_axis)
    return x_axis, y_axis, vary_keys


def _format_slice_title(slice_params: dict[str, Any]) -> str:
    if not slice_params:
        return ""
    parts = []
    for key, value in slice_params.items():
        label = _format_axis_value(key, value)
        parts.append(f"{key}={label}")
    return ", ".join(parts)


def _format_axis_value(key: str, value: Any) -> str:
    if value is None:
        return "base model" if key == "lora_path" else "none"
    if key == "lora_path":
        if isinstance(value, str):
            from pathlib import Path

            return Path(value).stem
        return str(value)
    text = str(value)
    if len(text) > 40:
        return text[:37] + "..."
    return text


def plan_grids(parameters: SweepParameters, grid: GridLayout) -> list[GridPlan]:
    combinations = build_combinations(parameters)
    if not combinations:
        return []

    vary_keys = parameters.vary_keys_with_values()
    if len(vary_keys) == 0:
        return []

    if len(vary_keys) == 1:
        key = vary_keys[0]
        combos = build_combinations(parameters)
        if not combos:
            return []
        param = parameters.get_param(key)
        if key == "lora_path":
            _, labels = lora_axis_display(param)
            values: list[Any] = labels
        else:
            values = param.effective_values()
        cells: list[list[int | None]] = [[c.index for c in combos]]
        return [
            GridPlan(
                index=0,
                x_axis=key,
                y_axis=key,
                x_values=values,
                y_values=[None],
                cells=cells,
                title="",
            )
        ]

    x_axis, y_axis, _ = _resolve_axes(parameters, grid, vary_keys)
    slice_keys = [k for k in vary_keys if k not in (x_axis, y_axis)]

    if not slice_keys:
        return [_build_single_grid(combinations, parameters, grid, 0, {}, x_axis, y_axis)]

    slice_axes: list[tuple[str, list[Any]]] = []
    for key in SWEEP_PARAM_ORDER:
        if key not in slice_keys:
            continue
        param = parameters.get_param(key)
        slice_axes.append((key, param.effective_values()))

    plans: list[GridPlan] = []
    from itertools import product

    for grid_index, slice_combo in enumerate(product(*(vals for _, vals in slice_axes))):
        slice_params = {key: val for (key, _), val in zip(slice_axes, slice_combo, strict=True)}
        filtered = [
            c
            for c in combinations
            if all(c.params.get(k) == v for k, v in slice_params.items())
        ]
        if not filtered:
            continue
        plans.append(
            _build_single_grid(filtered, parameters, grid, grid_index, slice_params, x_axis, y_axis)
        )
    return plans


def _build_single_grid(
    combinations: list[SweepCombination],
    parameters: SweepParameters,
    grid: GridLayout,
    index: int,
    slice_params: dict[str, Any],
    x_axis: str,
    y_axis: str,
) -> GridPlan:
    x_param = parameters.get_param(x_axis)
    y_param = parameters.get_param(y_axis)
    if x_axis in parameters.vary_keys_with_values():
        if x_axis == "lora_path":
            x_keys, x_values = lora_axis_display(x_param)
        else:
            x_keys = x_param.effective_values()
            x_values = list(x_keys)
    else:
        x_keys, x_values = [], [None]
    if y_axis in parameters.vary_keys_with_values():
        if y_axis == "lora_path":
            y_keys, y_values = lora_axis_display(y_param)
        else:
            y_keys = y_param.effective_values()
            y_values = list(y_keys)
    else:
        y_keys, y_values = [], [None]

    if x_axis == y_axis:
        if x_axis == "lora_path":
            x_keys, x_values = lora_axis_display(x_param)
        else:
            x_keys = x_param.effective_values()
            x_values = list(x_keys)
        y_keys = [None]
        y_values = [None]
    elif not x_keys:
        x_keys, x_values = [None], [None]
    if not y_keys:
        y_keys, y_values = [None], [None]

    index_map: dict[tuple[Any, Any], int] = {}
    for combo in combinations:
        x_val = combo.params.get(x_axis)
        y_val = combo.params.get(y_axis) if x_axis != y_axis else None
        index_map[(x_val, y_val)] = combo.index

    cells: list[list[int | None]] = []
    for y_val in y_keys:
        row: list[int | None] = []
        for x_val in x_keys:
            row.append(index_map.get((x_val, y_val)))
        cells.append(row)

    return GridPlan(
        index=index,
        slice_params=slice_params,
        x_axis=x_axis,
        y_axis=y_axis,
        x_values=x_values,
        y_values=y_values,
        cells=cells,
        title=_format_slice_title(slice_params),
    )
