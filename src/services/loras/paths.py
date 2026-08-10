"""Resolve filesystem paths for LoRA artifacts."""

from dataclasses import dataclass
from pathlib import Path

from src.db.tables.lora import Lora
from src.services.loras.weights import pick_weights_file
from src.storage.paths import StorageKind, StoragePaths
from src.trainer.config import TrainConfig


@dataclass(frozen=True)
class LoraPaths:
    name: str
    base_model_name: str
    weights_path: Path
    work_dir: Path
    relative_path: str
    weights_relpath: str


def runtime_train_config(lora: Lora) -> TrainConfig:
    if not lora.config_yaml:
        raise ValueError(f"LoRA id={lora.id} has no config_yaml")
    return TrainConfig.from_snapshot_yaml(lora.config_yaml)


def lora_work_dir_exists(relative_path: str) -> bool:
    if not relative_path or not StoragePaths.is_managed_relative_path(StorageKind.LORA, relative_path):
        return False
    try:
        return StoragePaths.resolve(StorageKind.LORA, relative_path).is_dir()
    except (ValueError, OSError):
        return False


def resolve_work_dir(lora: Lora) -> Path:
    return StoragePaths.resolve(StorageKind.LORA, lora.relative_path)


def resolve_sample_base_dir(lora: Lora) -> Path:
    """Base directory for sample files — prefers active run output_path."""
    if lora.output_path:
        return Path(lora.output_path)
    return resolve_work_dir(lora)


def resolve_weights_path(lora: Lora) -> Path:
    path = Path(lora.weights_relpath)
    if path.is_absolute():
        return path.expanduser().resolve()
    return StoragePaths.resolve(StorageKind.LORA, lora.weights_relpath)


def lora_artifacts_exist(lora: Lora) -> bool:
    if not lora_work_dir_exists(lora.relative_path):
        return False
    try:
        weights = resolve_weights_path(lora)
    except (ValueError, OSError):
        return False
    return weights.is_file()


def build_lora_paths(
    *,
    work_dir: Path,
    name: str,
    base_model_name: str,
    weights_path: Path | None = None,
) -> LoraPaths | None:
    relative_path = StoragePaths.to_relative(StorageKind.LORA, work_dir)
    if relative_path is None:
        return None
    weights = weights_path or pick_weights_file(work_dir)
    if weights is None:
        return None
    weights_relpath = StoragePaths.to_relative(StorageKind.LORA, weights)
    if weights_relpath is None:
        return None
    return LoraPaths(
        name=name,
        base_model_name=base_model_name,
        weights_path=weights,
        work_dir=work_dir,
        relative_path=relative_path,
        weights_relpath=weights_relpath,
    )


def resolve_completed_lora_paths(lora: Lora) -> LoraPaths | None:
    config = runtime_train_config(lora)
    if lora.output_path:
        work_dir = Path(lora.output_path)
    else:
        work_dir = StoragePaths.resolve_training_work_dir(config.output_dir, config.lora_name)
    return build_lora_paths(work_dir=work_dir, name=lora.name, base_model_name=config.base_model_name)
