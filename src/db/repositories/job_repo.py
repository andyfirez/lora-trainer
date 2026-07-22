"""Repository for unified Job table."""

from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.job import Job, JobStatus, JobType


class JobRepository(BaseRepository[Job]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Job, session)

    async def get_by_status(self, status: JobStatus) -> Sequence[Job]:
        result = await self._exec(
            select(Job).where(Job.status == status).order_by(Job.created_at)
        )
        return result.all()

    async def get_by_type(self, job_type: JobType) -> Sequence[Job]:
        result = await self._exec(
            select(Job).where(Job.job_type == job_type).order_by(Job.created_at)
        )
        return result.all()

    async def get_by_source_job_id(self, source_job_id: int) -> Sequence[Job]:
        result = await self._exec(
            select(Job)
            .where(Job.source_job_id == source_job_id)
            .order_by(Job.created_at.desc())
        )
        return result.all()

    async def get_running(self) -> Optional[Job]:
        result = await self._exec(
            select(Job).where(Job.status == JobStatus.RUNNING).limit(1)
        )
        return result.first()

    async def update_status(
        self,
        job: Job,
        status: JobStatus,
        *,
        pid: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Job:
        old_status = job.status
        now = datetime.now(timezone.utc)
        if old_status == JobStatus.RUNNING and status != JobStatus.RUNNING:
            if job.running_started_at is not None:
                started_at = job.running_started_at
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                job.accumulated_elapsed_seconds += (now - started_at).total_seconds()
                job.running_started_at = None
        elif old_status != JobStatus.RUNNING and status == JobStatus.RUNNING:
            job.running_started_at = now

        job.status = status
        job.updated_at = now
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
        job: Job,
        step: int,
        total: int,
        *,
        loss: Optional[float] = None,
        avr_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        epoch_total: Optional[int] = None,
    ) -> Job:
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

    async def update_cache_progress(self, job: Job, step: int, total: int) -> Job:
        job.cache_progress_step = step
        job.cache_progress_total = total
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_sampling_status(self, job: Job, status: Optional[str]) -> Job:
        job.sampling_status = status
        if status is None:
            job.sampling_step = None
            job.sampling_total = None
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_sampling_progress(self, job: Job, step: int, total: int) -> Job:
        job.sampling_step = step
        job.sampling_total = total
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_progress_status(self, job: Job, status: Optional[str]) -> Job:
        job.progress_status = status
        if status is None:
            job.progress_step = None
            job.progress_total = None
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_log_path(self, job: Job, log_path: str) -> Job:
        job.log_path = log_path
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_output_path(self, job: Job, output_path: str) -> Job:
        job.output_path = output_path
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def update_last_checkpoint(
        self,
        job: Job,
        *,
        checkpoint_path: str,
        epoch: int,
        step: int,
    ) -> Job:
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
        job: Job,
        *,
        checkpoint_path: str,
        epoch: int,
        step: int,
    ) -> Job:
        job.resume_checkpoint_path = checkpoint_path
        job.resume_from_epoch = epoch
        job.resume_from_step = step
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def clear_resume_state(self, job: Job) -> Job:
        job.resume_checkpoint_path = None
        job.resume_from_epoch = None
        job.resume_from_step = None
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def request_checkpoint_save(self, job: Job, requested: bool) -> Job:
        job.save_checkpoint_requested = requested
        job.updated_at = datetime.now(timezone.utc)
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def clear_runtime_state(self, job: Job) -> Job:
        job.pid = None
        job.error_message = None
        job.running_started_at = None
        job.accumulated_elapsed_seconds = 0.0
        if job.job_type == JobType.TRAINING:
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
        elif job.job_type in (JobType.SAMPLING, JobType.TAGGING):
            job.progress_step = None
            job.progress_total = None
            await self.update_progress_status(job, None)
        else:
            await self.update_progress_status(job, None)
        return job

    async def clear_process_state(self, job: Job) -> Job:
        job.pid = None
        if job.job_type == JobType.TRAINING:
            job.save_checkpoint_requested = False
            await self.update_sampling_status(job, None)
        elif job.job_type in (JobType.SAMPLING, JobType.TAGGING):
            job.progress_step = None
            job.progress_total = None
            await self.update_progress_status(job, None)
        else:
            await self.update_progress_status(job, None)
        return job
