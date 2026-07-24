"""Normalize legacy training config YAML stored in the database."""

from __future__ import annotations

from typing import Any

import yaml

from src.trainer.config import (
    FORBIDDEN_DEPRECATED_TRAIN_KEYS,
    RUNTIME_SAMPLING_FIELDS,
    TrainConfig,
)


def _migrate_learning_rate(data: dict[str, Any]) -> None:
    legacy_lr = data.pop("learning_rate", None)
    if legacy_lr is None:
        return
    for part in ("unet", "text_encoder_1", "text_encoder_2"):
        part_data = data.get(part)
        if part_data is None:
            data[part] = {"learning_rate": legacy_lr}
        elif isinstance(part_data, dict) and "learning_rate" not in part_data:
            part_data["learning_rate"] = legacy_lr


def _strip_deprecated_training_keys(data: dict[str, Any]) -> None:
    for key in FORBIDDEN_DEPRECATED_TRAIN_KEYS:
        data.pop(key, None)
    for key in RUNTIME_SAMPLING_FIELDS:
        data.pop(key, None)


def migrate_training_yaml_raw(data: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(data)
    _migrate_learning_rate(migrated)
    _strip_deprecated_training_keys(migrated)
    return migrated


def migrate_training_yaml(config_yaml: str) -> str | None:
    """Return migrated YAML, or None if unchanged or invalid."""
    if not config_yaml:
        return None
    try:
        data = yaml.safe_load(config_yaml) or {}
        if not isinstance(data, dict):
            return None
        migrated_data = migrate_training_yaml_raw(data)
        prepared = yaml.dump(migrated_data, allow_unicode=True, sort_keys=False)
        config = TrainConfig.from_yaml(prepared)
        migrated = config.to_yaml()
        if yaml.safe_load(migrated) == yaml.safe_load(config_yaml):
            return None
        return migrated
    except Exception:
        return None
