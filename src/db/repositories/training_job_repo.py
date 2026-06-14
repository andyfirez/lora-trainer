"""Repository for TrainingJob table."""

from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.training_job import JobStatus, TrainingJob


class TrainingJobRepository(BaseRepository[TrainingJob]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(TrainingJob, session)

    async def get_by_status(self, status: JobStatus) -> Sequence[TrainingJob]:
        result = await self._exec(
            select(TrainingJob).where(TrainingJob.status == status).order_by(TrainingJob.created_at)
        )
        return result.all()

    async def get_running(self) -> Optional[TrainingJob]:
        result = await self._exec(
            select(TrainingJob).where(TrainingJob.status == JobStatus.RUNNING).limit(1)
        )
        return result.first()

    async def update_status(
        self,
        job: TrainingJob,
        status: JobStatus,
        *,
        pid: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> TrainingJob:
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        if pid is not None:
            job.pid = pid
        if error_message is not None:
            job.error_message = error_message
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_progress(
        self,
        job: TrainingJob,
        step: int,
        total: int,
        *,
        loss: Optional[float] = None,
        avr_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        epoch_total: Optional[int] = None,
    ) -> TrainingJob:
        job.progress_step = step
        job.progress_total = total
        if loss is not None:
            job.progress_loss = loss
        if avr_loss is not None:
            job.progress_avr_loss = avr_loss
        if epoch is not None:
            job.progress_epoch = epoch
        if epoch_total is not None:
            job.progress_epoch_total = epoch_total
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_cache_progress(
        self,
        job: TrainingJob,
        step: int,
        total: int,
    ) -> TrainingJob:
        job.cache_progress_step = step
        job.cache_progress_total = total
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_sampling_status(self, job: TrainingJob, status: Optional[str]) -> TrainingJob:
        job.sampling_status = status
        if status is None:
            job.sampling_step = None
            job.sampling_total = None
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_sampling_progress(self, job: TrainingJob, step: int, total: int) -> TrainingJob:
        job.sampling_step = step
        job.sampling_total = total
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_log_path(self, job: TrainingJob, log_path: str) -> TrainingJob:
        job.log_path = log_path
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_output_path(self, job: TrainingJob, output_path: str) -> TrainingJob:
        job.output_path = output_path
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_last_checkpoint(
        self,
        job: TrainingJob,
        *,
        checkpoint_path: str,
        epoch: int,
        step: int,
    ) -> TrainingJob:
        job.last_checkpoint_path = checkpoint_path
        job.last_checkpoint_epoch = epoch
        job.last_checkpoint_step = step
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def set_resume_state(
        self,
        job: TrainingJob,
        *,
        checkpoint_path: str,
        epoch: int,
        step: int,
    ) -> TrainingJob:
        job.resume_checkpoint_path = checkpoint_path
        job.resume_from_epoch = epoch
        job.resume_from_step = step
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def clear_resume_state(self, job: TrainingJob) -> TrainingJob:
        job.resume_checkpoint_path = None
        job.resume_from_epoch = None
        job.resume_from_step = None
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def request_checkpoint_save(self, job: TrainingJob, requested: bool) -> TrainingJob:
        job.save_checkpoint_requested = requested
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def clear_runtime_state(self, job: TrainingJob) -> TrainingJob:
        job.pid = None
        job.error_message = None
        job.progress_step = None
        job.progress_total = None
        job.progress_loss = None
        job.progress_avr_loss = None
        job.progress_epoch = None
        job.progress_epoch_total = None
        job.cache_progress_step = None
        job.cache_progress_total = None
        job.save_checkpoint_requested = False
        await self.update_sampling_status(job, None)
        return job

    async def clear_process_state(self, job: TrainingJob) -> TrainingJob:
        job.pid = None
        job.save_checkpoint_requested = False
        await self.update_sampling_status(job, None)
        return job
