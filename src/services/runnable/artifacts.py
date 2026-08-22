"""Shared log/sample artifact helpers for Runnable entities (Lora, Sampling)."""

from pathlib import Path
from typing import Protocol

from src.services.runnable.samples import list_samples_for_output_dir
from src.trainer.training_log import JobTrainingLogger


class _RunnableWithArtifacts(Protocol):
    log_path: str | None
    output_path: str | None


def read_runnable_logs(entity: _RunnableWithArtifacts, tail: int = 500) -> list[str]:
    if not entity.log_path:
        return []
    return JobTrainingLogger.read_tail(Path(entity.log_path), lines=tail)


def list_runnable_samples(entity: _RunnableWithArtifacts) -> list[tuple[Path, str, dict]]:
    if not entity.output_path:
        return []
    sampling_id = getattr(entity, "id", None)
    return list_samples_for_output_dir(Path(entity.output_path), sampling_id=sampling_id)
