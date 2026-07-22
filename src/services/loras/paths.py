"""Resolve filesystem paths for trained LoRA artifacts."""

from dataclasses import dataclass
from pathlib import Path

from src.db.tables.job import Job
from src.trainer.config import TrainConfig


@dataclass(frozen=True)
class TrainedLoraPaths:
    name: str
    base_model_name: str
    weights_path: Path
    work_dir: Path


def runtime_train_config(job: Job) -> TrainConfig:
    return TrainConfig.from_yaml(job.config_yaml)


def resolve_trained_lora_paths(job: Job) -> TrainedLoraPaths | None:
    config = runtime_train_config(job)
    work_dir = Path(job.output_path) if job.output_path else Path(config.output_dir) / config.lora_name
    ext = config.output_format.value
    weights_path = work_dir / f"{config.lora_name}.{ext}"
    if not weights_path.is_file():
        return None
    return TrainedLoraPaths(
        name=config.lora_name,
        base_model_name=config.base_model_name,
        weights_path=weights_path,
        work_dir=work_dir,
    )


def unique_lora_name(base_name: str, job_id: int) -> str:
    return f"{base_name}_j{job_id}"


def assign_unique_training_job_yaml(config_yaml: str, job_id: int) -> str:
    from src.services.configs.versioning import strip_lora_version_suffix

    config = TrainConfig.from_yaml(config_yaml)
    base_name = strip_lora_version_suffix(config.lora_name)
    return config.model_copy(update={"lora_name": unique_lora_name(base_name, job_id)}).to_yaml()
