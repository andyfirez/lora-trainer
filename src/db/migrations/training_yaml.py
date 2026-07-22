"""Normalize legacy training config YAML stored in the database."""

from __future__ import annotations

import yaml

from src.trainer.config import TrainConfig


def migrate_training_yaml(config_yaml: str) -> str | None:
    """Return migrated YAML, or None if unchanged or invalid."""
    if not config_yaml:
        return None
    try:
        data = yaml.safe_load(config_yaml) or {}
        if not isinstance(data, dict):
            return None
        data.pop("sample_after_training", None)
        prepared = yaml.dump(data, allow_unicode=True, sort_keys=False)
        config = TrainConfig.from_yaml(prepared)
        migrated = config.to_yaml()
        return migrated if migrated != config_yaml else None
    except Exception:
        return None
