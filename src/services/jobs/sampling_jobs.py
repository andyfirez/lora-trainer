"""Sampling job validation and path resolution helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.db.repositories.job_repo import JobRepository
from src.db.tables.job import Job, JobType
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.services.sampling.exceptions import (
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)
from src.trainer.config import TrainConfig


def validate_sample_prompts(config: SamplingConfig) -> None:
    if not config.effective_prompts():
        raise SamplingPromptsNotConfiguredError()


def validate_lora_paths(lora_paths: list[str]) -> None:
    for lora_path in lora_paths:
        path = Path(lora_path)
        if not path.is_file():
            raise SamplingLoRAPathNotFoundError(lora_path)


def resolve_lora_paths_from_sampling_config(sampling_config: SamplingConfig) -> list[str]:
    """Collect LoRA paths stored in a sampling config (manual source)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(path: object | None) -> None:
        if path is None:
            return
        text = str(path).strip()
        if not text or text in seen:
            return
        seen.add(text)
        ordered.append(text)

    for path in sampling_config.lora_paths:
        add(path)

    lora_param = sampling_config.parameters.lora_path
    for value in lora_param.effective_values():
        add(value)

    return ordered


def prepare_sampling_config_lora_paths(
    sampling_config: SamplingConfig,
    job_lora_paths: list[str] | None = None,
) -> tuple[SamplingConfig, list[str]]:
    """Merge job-level and config-level LoRA paths; ensure parameters are populated."""
    paths: list[str] = []
    seen: set[str] = set()
    for candidate in (job_lora_paths or []) + resolve_lora_paths_from_sampling_config(sampling_config):
        text = str(candidate).strip()
        if text and text not in seen:
            seen.add(text)
            paths.append(text)
    if not paths:
        return sampling_config, []
    return sampling_config.with_resolved_lora_paths(paths), paths


def find_intermediate_checkpoints(config: TrainConfig) -> list[Path]:
    work_dir = Path(config.output_dir) / config.lora_name
    ext = config.output_format.value
    epoch_paths = list(work_dir.glob(f"{config.lora_name}_epoch*.{ext}"))
    step_paths = list(work_dir.glob(f"{config.lora_name}_step*.{ext}"))
    return sorted(epoch_paths + step_paths, key=lambda path: (path.stat().st_mtime, path.name))


async def resolve_sampling_lora_paths(
    job_repo: JobRepository,
    source_job_id: int | None,
    *,
    runtime_train_config: Callable[[Job], TrainConfig],
) -> list[str]:
    if source_job_id is None:
        return []
    source_job = await job_repo.get_by_id(source_job_id)
    if source_job is None or source_job.job_type != JobType.TRAINING:
        return []
    train_config = runtime_train_config(source_job)
    return [str(path) for path in find_intermediate_checkpoints(train_config)]


async def resolve_sampling_output_dir(
    job_repo: JobRepository,
    sampling_config: SamplingConfig,
    job_id: int,
    source_job_id: int | None,
    *,
    runtime_train_config: Callable[[Job], TrainConfig],
) -> Path:
    source_train_config: TrainConfig | None = None
    if source_job_id is not None:
        source_job = await job_repo.get_by_id(source_job_id)
        if source_job is not None and source_job.job_type == JobType.TRAINING:
            source_train_config = runtime_train_config(source_job)
    return resolve_sampling_output_path(sampling_config, job_id, source_train_config)
