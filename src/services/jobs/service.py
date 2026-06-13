"""Business logic for training jobs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.queue_entry import QueueEntry
from src.db.tables.training_job import JobStatus, TrainingJob
from src.services.jobs.exceptions import (
    JobAlreadyQueuedError,
    JobNotCancellableError,
    JobNotFoundError,
)
from src.trainer.training_log import JobTrainingLogger


class JobsService:
    def __init__(
        self,
        job_repo: TrainingJobRepository,
        queue_repo: QueueRepository,
    ) -> None:
        self._job_repo = job_repo
        self._queue_repo = queue_repo

    async def list_jobs(self) -> Sequence[TrainingJob]:
        return await self._job_repo.get_all()

    async def get_job(self, job_id: int) -> TrainingJob:
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    async def create_job(self, name: str, config_yaml: str) -> TrainingJob:
        job = TrainingJob(name=name, config_yaml=config_yaml)
        return await self._job_repo.add(job)

    async def update_job(self, job_id: int, name: Optional[str], config_yaml: Optional[str]) -> TrainingJob:
        job = await self.get_job(job_id)
        if name is not None:
            job.name = name
        if config_yaml is not None:
            job.config_yaml = config_yaml
        job.updated_at = datetime.now(timezone.utc)
        self._job_repo._session.add(job)
        await self._job_repo._session.flush()
        await self._job_repo._session.refresh(job)
        return job

    async def delete_job(self, job_id: int) -> None:
        job = await self.get_job(job_id)
        entry = await self._queue_repo.get_by_job_id(job_id)
        if entry is not None:
            await self._queue_repo.delete(entry)
        await self._job_repo.delete(job)

    async def enqueue_job(self, job_id: int) -> QueueEntry:
        job = await self.get_job(job_id)
        existing = await self._queue_repo.get_by_job_id(job_id)
        if existing is not None:
            raise JobAlreadyQueuedError(job_id)
        max_pos = await self._queue_repo.get_max_position()
        entry = QueueEntry(job_id=job_id, position=max_pos + 1)
        await self._job_repo.clear_runtime_state(job)
        await self._job_repo.update_status(job, JobStatus.QUEUED)
        return await self._queue_repo.add(entry)

    async def cancel_job(self, job_id: int) -> TrainingJob:
        job = await self.get_job(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise JobNotCancellableError(job_id, job.status)
        if job.status == JobStatus.RUNNING:
            await self._job_repo.update_status(job, JobStatus.CANCELLED)
            return await self._job_repo.clear_runtime_state(job)
        entry = await self._queue_repo.get_by_job_id(job_id)
        if entry is not None:
            await self._queue_repo.shift_positions_down(entry.position)
            await self._queue_repo.delete(entry)
        await self._job_repo.update_status(job, JobStatus.CANCELLED)
        return await self._job_repo.clear_runtime_state(job)

    async def get_job_logs(self, job_id: int, tail: int = 500) -> list[str]:
        job = await self.get_job(job_id)
        if not job.log_path:
            return []
        return JobTrainingLogger.read_tail(Path(job.log_path), lines=tail)
