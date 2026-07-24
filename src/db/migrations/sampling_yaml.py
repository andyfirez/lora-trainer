"""Normalize legacy sampling config YAML stored in the database."""

from __future__ import annotations

from typing import Any

import yaml

from src.sampler.config import (
    FORBIDDEN_DEPRECATED_SAMPLING_KEYS,
    LEGACY_FLAT_SAMPLING_KEYS,
    SamplingConfig,
)
from src.sampler.sweep.models import (
    SweepMode,
    SweepParameter,
    SweepParameters,
    lora_entry_to_param_value,
    parse_lora_entry,
)

_LEGACY_TO_PARAM: dict[str, str] = {
    "base_model_name": "base_model_name",
    "sample_negative_prompt": "negative_prompt",
    "sample_steps": "steps",
    "sample_cfg_scale": "cfg_scale",
    "sample_width": "width",
    "sample_height": "height",
    "sample_scheduler": "scheduler",
}


def _has_parameters_block(data: dict[str, Any]) -> bool:
    parameters = data.get("parameters")
    return isinstance(parameters, dict) and bool(parameters)


def _normalize_lora_path_parameter(params: SweepParameters) -> SweepParameters:
    lora_param = params.lora_path
    if lora_param.mode == SweepMode.VARY and lora_param.values:
        return params.model_copy(
            update={
                "lora_path": SweepParameter(
                    mode=SweepMode.VARY,
                    values=[lora_entry_to_param_value(parse_lora_entry(value)) for value in lora_param.values],
                )
            }
        )
    if lora_param.value is not None:
        return params.model_copy(
            update={
                "lora_path": SweepParameter(
                    mode=SweepMode.FIXED,
                    value=lora_entry_to_param_value(parse_lora_entry(lora_param.value)),
                )
            }
        )
    return params


def _migrate_legacy_into_parameters(data: dict[str, Any], params: SweepParameters) -> SweepParameters:
    migrated = params
    prompts = data.get("sample_prompts") or []
    if isinstance(prompts, list) and prompts:
        if len(prompts) > 1:
            migrated = migrated.model_copy(
                update={"prompt": SweepParameter(mode=SweepMode.VARY, values=list(prompts))}
            )
        else:
            migrated = migrated.model_copy(
                update={"prompt": SweepParameter(mode=SweepMode.FIXED, value=prompts[0])}
            )

    for legacy_key, param_key in _LEGACY_TO_PARAM.items():
        if legacy_key not in data:
            continue
        migrated = migrated.model_copy(
            update={param_key: SweepParameter(mode=SweepMode.FIXED, value=data[legacy_key])}
        )

    lora_paths = data.get("lora_paths")
    if isinstance(lora_paths, list) and lora_paths and not migrated.lora_path.effective_values():
        entries = [lora_entry_to_param_value(parse_lora_entry(path)) for path in lora_paths]
        if len(entries) == 1:
            migrated = migrated.model_copy(
                update={"lora_path": SweepParameter(mode=SweepMode.FIXED, value=entries[0])}
            )
        else:
            migrated = migrated.model_copy(
                update={"lora_path": SweepParameter(mode=SweepMode.VARY, values=entries)}
            )

    return _normalize_lora_path_parameter(migrated)


def _lora_paths_from_parameters(params: SweepParameters) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for value in params.lora_path.effective_values():
        path = parse_lora_entry(value).path
        if path is None:
            continue
        normalized = str(path).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
    return paths


def migrate_sampling_yaml_raw(data: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(data)

    for key in FORBIDDEN_DEPRECATED_SAMPLING_KEYS:
        migrated.pop(key, None)

    if _has_parameters_block(migrated):
        params = SweepParameters.model_validate(migrated["parameters"])
        params = _normalize_lora_path_parameter(params)
    else:
        params = _migrate_legacy_into_parameters(migrated, SweepParameters())

    for key in LEGACY_FLAT_SAMPLING_KEYS:
        migrated.pop(key, None)

    migrated["parameters"] = params.model_dump(mode="json")
    file_paths = _lora_paths_from_parameters(params)
    if file_paths:
        migrated["lora_paths"] = file_paths
    elif "lora_paths" in migrated and not file_paths:
        migrated["lora_paths"] = []

    return migrated


def migrate_sampling_yaml(config_yaml: str) -> str | None:
    """Return migrated YAML, or None if unchanged or invalid."""
    if not config_yaml:
        return None
    try:
        data = yaml.safe_load(config_yaml) or {}
        if not isinstance(data, dict):
            return None
        migrated_data = migrate_sampling_yaml_raw(data)
        normalized = SamplingConfig.model_validate(migrated_data).to_yaml()
        if yaml.safe_load(normalized) == yaml.safe_load(config_yaml):
            return None
        return normalized
    except Exception:
        return None
