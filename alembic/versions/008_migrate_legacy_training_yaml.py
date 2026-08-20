"""Migrate legacy training config YAML (learning_rate, inline sampling fields)."""

from typing import Any, Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op
from src.trainer.config import (
    FORBIDDEN_DEPRECATED_TRAIN_KEYS,
    RUNTIME_SAMPLING_FIELDS,
    TrainConfig,
)
from src.trainer.gpu_resolution import strip_gpu_overrides_matching_defaults

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TRAINING_YAML_TABLES: tuple[tuple[str, str | None], ...] = (
    ("job_configs", "config_type = 'training'"),
    ("jobs", "job_type = 'training'"),
    ("trained_loras", None),
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
    data.pop("tf32", None)
    data.pop("attention_mechanism", None)


def _normalize_gpu_overrides(data: dict[str, Any]) -> None:
    from src.settings.app_settings import settings

    normalized = strip_gpu_overrides_matching_defaults(data, settings.gpu_defaults)
    for key in list(data.keys()):
        if key not in normalized:
            data.pop(key, None)


def _migrate_training_yaml(config_yaml: str) -> str | None:
    if not config_yaml:
        return None
    try:
        data = yaml.safe_load(config_yaml) or {}
        if not isinstance(data, dict):
            return None
        migrated_data = dict(data)
        _migrate_learning_rate(migrated_data)
        _strip_deprecated_training_keys(migrated_data)
        _normalize_gpu_overrides(migrated_data)
        prepared = yaml.dump(migrated_data, allow_unicode=True, sort_keys=False)
        config = TrainConfig.from_yaml(prepared)
        migrated = config.to_yaml()
        if yaml.safe_load(migrated) == yaml.safe_load(config_yaml):
            return None
        return migrated
    except Exception:
        return None


def _migrate_table(connection: sa.Connection, table: str, where_clause: str | None) -> None:
    query = f"SELECT id, config_yaml FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    rows = connection.execute(sa.text(query)).fetchall()
    for row_id, config_yaml in rows:
        if not config_yaml:
            continue
        migrated = _migrate_training_yaml(config_yaml)
        if migrated is None:
            continue
        connection.execute(
            sa.text(f"UPDATE {table} SET config_yaml = :config_yaml WHERE id = :id"),
            {"config_yaml": migrated, "id": row_id},
        )


def upgrade() -> None:
    connection = op.get_bind()
    for table, where_clause in _TRAINING_YAML_TABLES:
        _migrate_table(connection, table, where_clause)


def downgrade() -> None:
    pass
