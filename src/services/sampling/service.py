"""Business logic for standalone sampling runs."""

from pathlib import Path
from typing import Sequence

import yaml

from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.sampling_run_repo import SamplingRunRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.queue_entry import QueueEntry, QueueItemType
from src.db.tables.sampling_run import SamplingRun, SamplingRunStatus
from src.db.tables.training_job import TrainingJob
from src.services.jobs.exceptions import JobNotFoundError
from src.services.sampling.exceptions import (
    SamplingCheckpointsNotFoundError,
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
    SamplingRunAlreadyQueuedError,
    SamplingRunNotCancellableError,
    SamplingRunNotFoundError,
)
from src.trainer.config import TrainConfig
from src.trainer.training_log import JobTrainingLogger


class SamplingService:
    def __init__(
        self,
        sampling_run_repo: SamplingRunRepository,
        queue_repo: QueueRepository,
        job_repo: TrainingJobRepository,
    ) -> None:
        self._sampling_run_repo = sampling_run_repo
        self._queue_repo = queue_repo
        self._job_repo = job_repo

    async def list_runs(self, *, source_job_id: int | None = None) -> Sequence[SamplingRun]:
        if source_job_id is not None:
            return await self._sampling_run_repo.get_by_source_job_id(source_job_id)
        return await self._sampling_run_repo.get_all()

    async def get_run(self, sampling_run_id: int) -> SamplingRun:
        sampling_run = await self._sampling_run_repo.get_by_id(sampling_run_id)
        if sampling_run is None:
            raise SamplingRunNotFoundError(sampling_run_id)
        return sampling_run

    async def create_run(
        self,
        *,
        name: str | None,
        config_yaml: str,
        lora_paths: list[str],
        source_job_id: int | None = None,
    ) -> SamplingRun:
        self._validate_lora_paths(lora_paths)
        if source_job_id is not None and await self._job_repo.get_by_id(source_job_id) is None:
            raise JobNotFoundError(source_job_id)
        config = TrainConfig.from_yaml(config_yaml)
        self._validate_sample_prompts(config)
        sampling_run = SamplingRun(
            name=name or self._default_run_name(lora_paths),
            config_yaml=config_yaml,
            lora_paths_yaml=yaml.safe_dump(lora_paths, allow_unicode=True, sort_keys=False),
            source_job_id=source_job_id,
        )
        sampling_run = await self._sampling_run_repo.add(sampling_run)
        await self._sampling_run_repo.update_output_path(
            sampling_run,
            str(self._default_output_dir(config, sampling_run.id)),
        )
        return sampling_run

    async def create_for_job(
        self,
        job_id: int,
        *,
        name: str | None,
        lora_paths: list[str],
    ) -> SamplingRun:
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.id is None:
            raise JobNotFoundError(job_id)
        return await self.create_run(
            name=name or f"{job.name} sampling",
            config_yaml=job.config_yaml,
            lora_paths=lora_paths,
            source_job_id=job.id,
        )

    async def create_auto_run_for_job(self, job: TrainingJob) -> SamplingRun | None:
        if job.id is None:
            return None
        config = TrainConfig.from_yaml(job.config_yaml)
        if not config.sample_after_training:
            return None
        checkpoint_paths = self._find_intermediate_checkpoints(config)
        if not checkpoint_paths:
            raise SamplingCheckpointsNotFoundError(job.id)
        sampling_run = await self.create_run(
            name=f"{job.name} post-train sampling",
            config_yaml=job.config_yaml,
            lora_paths=[str(path) for path in checkpoint_paths],
            source_job_id=job.id,
        )
        await self.enqueue_run(sampling_run.id)
        return sampling_run

    async def enqueue_run(self, sampling_run_id: int) -> QueueEntry:
        sampling_run = await self.get_run(sampling_run_id)
        existing = await self._queue_repo.get_by_sampling_run_id(sampling_run.id)
        if existing is not None:
            raise SamplingRunAlreadyQueuedError(sampling_run.id)
        await self._sampling_run_repo.clear_runtime_state(sampling_run)
        max_pos = await self._queue_repo.get_max_position()
        entry = QueueEntry(
            item_type=QueueItemType.SAMPLING,
            item_id=sampling_run.id,
            position=max_pos + 1,
        )
        await self._sampling_run_repo.update_status(sampling_run, SamplingRunStatus.QUEUED)
        return await self._queue_repo.add(entry)

    async def cancel_run(self, sampling_run_id: int) -> SamplingRun:
        sampling_run = await self.get_run(sampling_run_id)
        if sampling_run.status in (
            SamplingRunStatus.COMPLETED,
            SamplingRunStatus.FAILED,
            SamplingRunStatus.CANCELLED,
        ):
            raise SamplingRunNotCancellableError(sampling_run_id, sampling_run.status)
        if sampling_run.status == SamplingRunStatus.RUNNING:
            await self._sampling_run_repo.update_status(sampling_run, SamplingRunStatus.CANCELLED)
            return await self._sampling_run_repo.clear_process_state(sampling_run)
        entry = await self._queue_repo.get_by_sampling_run_id(sampling_run_id)
        if entry is not None:
            await self._queue_repo.shift_positions_down(entry.position)
            await self._queue_repo.delete(entry)
        await self._sampling_run_repo.update_status(sampling_run, SamplingRunStatus.CANCELLED)
        return await self._sampling_run_repo.clear_runtime_state(sampling_run)

    def get_lora_paths(self, sampling_run: SamplingRun) -> list[str]:
        data = yaml.safe_load(sampling_run.lora_paths_yaml) or []
        return [str(path) for path in data]

    async def get_run_logs(self, sampling_run_id: int, tail: int = 500) -> list[str]:
        sampling_run = await self.get_run(sampling_run_id)
        if not sampling_run.log_path:
            return []
        return JobTrainingLogger.read_tail(Path(sampling_run.log_path), lines=tail)

    def list_samples(self, sampling_run: SamplingRun) -> list[Path]:
        if not sampling_run.output_path:
            return []
        output_dir = Path(sampling_run.output_path)
        if not output_dir.exists():
            return []
        return sorted(output_dir.glob("*.png"))

    def _validate_sample_prompts(self, config: TrainConfig) -> None:
        if not config.sample_prompts:
            raise SamplingPromptsNotConfiguredError()

    def _validate_lora_paths(self, lora_paths: list[str]) -> None:
        for lora_path in lora_paths:
            path = Path(lora_path)
            if not path.is_file():
                raise SamplingLoRAPathNotFoundError(lora_path)

    def _find_intermediate_checkpoints(self, config: TrainConfig) -> list[Path]:
        work_dir = Path(config.output_dir) / config.lora_name
        ext = config.output_format.value
        epoch_paths = list(work_dir.glob(f"{config.lora_name}_epoch*.{ext}"))
        step_paths = list(work_dir.glob(f"{config.lora_name}_step*.{ext}"))
        return sorted(epoch_paths + step_paths, key=lambda path: (path.stat().st_mtime, path.name))

    def _default_run_name(self, lora_paths: list[str]) -> str:
        first = Path(lora_paths[0]).stem
        return f"{first} sampling" if len(lora_paths) == 1 else f"{first} + {len(lora_paths) - 1} sampling"

    def _default_output_dir(self, config: TrainConfig, sampling_run_id: int | None) -> Path:
        run_id = sampling_run_id if sampling_run_id is not None else "new"
        return Path(config.output_dir) / config.lora_name / "samples" / f"sampling_run_{run_id}"
