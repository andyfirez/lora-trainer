"""Resolve managed storage paths for training and sampling configs."""

from pathlib import Path

from src.storage.base_model_paths import resolve_config_base_model
from src.storage.paths import StoragePaths

__all__ = ["resolve_config_base_model", "resolve_config_output_dir", "resolve_config_lora_file"]


def resolve_config_output_dir(output_dir: str) -> Path:
    if Path(output_dir).is_absolute():
        return Path(output_dir).expanduser().resolve()
    return StoragePaths.resolve_lora_path(output_dir)


def resolve_config_lora_file(lora_path: str) -> Path:
    path = Path(lora_path)
    if path.is_absolute():
        return path.expanduser().resolve()
    return StoragePaths.resolve_lora_path(lora_path)
