"""Read and write config.toml for runtime settings updates."""

import os
import tomllib
from pathlib import Path

from src.settings.app_settings import settings
from src.settings.models import DatabaseSettings, ServerSettings, StorageSettings, TrainingSettings


def get_config_path() -> Path:
    return Path(os.environ.get("APP_CONFIG_FILE", "config.toml"))


def _format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f'"{value}"'


def _write_toml(path: Path, data: dict[str, object]) -> None:
    lines: list[str] = []
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_format_toml_value(value)}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _current_config_data() -> dict[str, dict[str, object]]:
    path = get_config_path()
    if path.is_file():
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
        return {key: dict(value) for key, value in loaded.items() if isinstance(value, dict)}

    return {
        "server": settings.server.model_dump(),
        "database": settings.database.model_dump(),
        "training": settings.training.model_dump(),
        "storage": settings.storage.model_dump(),
    }


def persist_training_settings(
    *,
    max_concurrent_jobs: int | None = None,
    worker_poll_interval_seconds: int | None = None,
) -> None:
    data = _current_config_data()
    training = dict(data.get("training", settings.training.model_dump()))
    if max_concurrent_jobs is not None:
        training["max_concurrent_jobs"] = max_concurrent_jobs
    if worker_poll_interval_seconds is not None:
        training["worker_poll_interval_seconds"] = worker_poll_interval_seconds
    data["training"] = training

    if "server" not in data:
        data["server"] = ServerSettings.model_validate(settings.server).model_dump()
    if "database" not in data:
        data["database"] = DatabaseSettings.model_validate(settings.database).model_dump()
    if "storage" not in data:
        data["storage"] = StorageSettings.model_validate(settings.storage).model_dump()

    _write_toml(get_config_path(), data)


def persist_storage_settings(
    *,
    datasets_root: str | None = None,
    base_models_root: str | None = None,
    lora_root: str | None = None,
) -> None:
    data = _current_config_data()
    storage = dict(data.get("storage", settings.storage.model_dump()))
    if datasets_root is not None:
        storage["datasets_root"] = datasets_root
    if base_models_root is not None:
        storage["base_models_root"] = base_models_root
    if lora_root is not None:
        storage["lora_root"] = lora_root
    data["storage"] = storage

    if "server" not in data:
        data["server"] = ServerSettings.model_validate(settings.server).model_dump()
    if "database" not in data:
        data["database"] = DatabaseSettings.model_validate(settings.database).model_dump()
    if "training" not in data:
        data["training"] = TrainingSettings.model_validate(settings.training).model_dump()

    _write_toml(get_config_path(), data)


def apply_storage_settings(
    *,
    datasets_root: str | None = None,
    base_models_root: str | None = None,
    lora_root: str | None = None,
) -> StorageSettings:
    updates: dict[str, str] = {}
    if datasets_root is not None:
        updates["datasets_root"] = datasets_root
    if base_models_root is not None:
        updates["base_models_root"] = base_models_root
    if lora_root is not None:
        updates["lora_root"] = lora_root
    if updates:
        settings.storage = settings.storage.model_copy(update=updates)
    return settings.storage


def apply_training_settings(
    *,
    max_concurrent_jobs: int | None = None,
    worker_poll_interval_seconds: int | None = None,
) -> TrainingSettings:
    updates: dict[str, int] = {}
    if max_concurrent_jobs is not None:
        updates["max_concurrent_jobs"] = max_concurrent_jobs
    if worker_poll_interval_seconds is not None:
        updates["worker_poll_interval_seconds"] = worker_poll_interval_seconds
    if updates:
        settings.training = settings.training.model_copy(update=updates)
    return settings.training
