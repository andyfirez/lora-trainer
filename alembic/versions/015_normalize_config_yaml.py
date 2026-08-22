"""Normalize legacy config YAML stored in job_configs, jobs, and trained_loras."""

from typing import Any, Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op
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
from src.trainer.config import (
    FORBIDDEN_DEPRECATED_TRAIN_KEYS,
    RUNTIME_SAMPLING_FIELDS,
    TrainConfig,
)
from src.trainer.gpu_resolution import strip_gpu_overrides_matching_defaults

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONFIG_YAML_TABLES: tuple[tuple[str, str | None, str], ...] = (
    ("job_configs", "config_type = 'sampling'", "sampling"),
    ("jobs", "job_type = 'sampling'", "sampling"),
    ("job_configs", "config_type = 'training'", "training"),
    ("jobs", "job_type = 'training'", "training"),
    ("trained_loras", None, "training"),
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


def _migrate_sampling_yaml_raw(data: dict[str, Any]) -> dict[str, Any]:
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

    migrated.pop("tf32", None)
    migrated.pop("attention_mechanism", None)
    from src.settings.app_settings import settings

    normalized = strip_gpu_overrides_matching_defaults(migrated, settings.gpu_defaults)
    for key in list(migrated.keys()):
        if key not in normalized and key in {"mixed_precision", "vae_dtype", "sample_vae_tiling"}:
            migrated.pop(key, None)

    return migrated


def _migrate_sampling_yaml(config_yaml: str) -> str | None:
    if not config_yaml:
        return None
    try:
        data = yaml.safe_load(config_yaml) or {}
        if not isinstance(data, dict):
            return None
        migrated_data = _migrate_sampling_yaml_raw(data)
        normalized = yaml.dump(
            SamplingConfig.model_validate(migrated_data)._entity_data(),
            allow_unicode=True,
            sort_keys=False,
        )
        if yaml.safe_load(normalized) == yaml.safe_load(config_yaml):
            return None
        return normalized
    except Exception:
        return None


def _migrate_table(connection: sa.Connection, table: str, where_clause: str | None, kind: str) -> None:
    query = f"SELECT id, config_yaml FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    rows = connection.execute(sa.text(query)).fetchall()
    migrate = _migrate_sampling_yaml if kind == "sampling" else _migrate_training_yaml
    for row_id, config_yaml in rows:
        if not config_yaml:
            continue
        migrated = migrate(config_yaml)
        if migrated is None:
            continue
        connection.execute(
            sa.text(f"UPDATE {table} SET config_yaml = :config_yaml WHERE id = :id"),
            {"config_yaml": migrated, "id": row_id},
        )


def upgrade() -> None:
    connection = op.get_bind()
    for table, where_clause, kind in _CONFIG_YAML_TABLES:
        _migrate_table(connection, table, where_clause, kind)


def downgrade() -> None:
    pass
