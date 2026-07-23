"""Resolve filesystem paths for trained LoRA artifacts."""

from dataclasses import dataclass
from pathlib import Path

from src.db.tables.job import Job
from src.db.tables.trained_lora import TrainedLora
from src.services.loras.weights import pick_weights_file
from src.storage.paths import StorageKind, StoragePaths
from src.trainer.config import TrainConfig


@dataclass(frozen=True)
class TrainedLoraPaths:
    name: str
    base_model_name: str
    weights_path: Path
    work_dir: Path
    relative_path: str
    weights_relpath: str


def runtime_train_config(job: Job) -> TrainConfig:
    return TrainConfig.from_yaml(job.config_yaml)


def lora_work_dir_exists(relative_path: str) -> bool:
    if not StoragePaths.is_managed_relative_path(StorageKind.LORA, relative_path):
        return False
    try:
        return StoragePaths.resolve(StorageKind.LORA, relative_path).is_dir()
    except (ValueError, OSError):
        return False


def resolve_work_dir(lora: TrainedLora) -> Path:
    return StoragePaths.resolve(StorageKind.LORA, lora.relative_path)


def resolve_weights_path(lora: TrainedLora) -> Path:
    path = Path(lora.weights_relpath)
    if path.is_absolute():
        return path.expanduser().resolve()
    return StoragePaths.resolve(StorageKind.LORA, lora.weights_relpath)


def lora_artifacts_exist(lora: TrainedLora) -> bool:
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
) -> TrainedLoraPaths | None:
    root = StoragePaths.lora_root()
    relative_path = StoragePaths.to_relative(StorageKind.LORA, work_dir)
    if relative_path is None:
        return None
    weights = weights_path or pick_weights_file(work_dir)
    if weights is None:
        return None
    weights_relpath = StoragePaths.to_relative(StorageKind.LORA, weights)
    if weights_relpath is None:
        return None
    return TrainedLoraPaths(
        name=name,
        base_model_name=base_model_name,
        weights_path=weights,
        work_dir=work_dir,
        relative_path=relative_path,
        weights_relpath=weights_relpath,
    )


def resolve_trained_lora_paths(job: Job) -> TrainedLoraPaths | None:
    config = runtime_train_config(job)
    if job.output_path:
        work_dir = Path(job.output_path)
    else:
        work_dir = StoragePaths.resolve_training_work_dir(config.output_dir, config.lora_name)
    return build_lora_paths(
        work_dir=work_dir,
        name=config.lora_name,
        base_model_name=config.base_model_name,
    )


def unique_lora_name(base_name: str, job_id: int) -> str:
    return f"{base_name}_j{job_id}"


def assign_unique_training_job_yaml(config_yaml: str, job_id: int) -> str:
    from src.services.configs.versioning import strip_lora_version_suffix

    config = TrainConfig.from_yaml(config_yaml)
    base_name = strip_lora_version_suffix(config.lora_name)
    return config.model_copy(update={"lora_name": unique_lora_name(base_name, job_id)}).to_yaml()
