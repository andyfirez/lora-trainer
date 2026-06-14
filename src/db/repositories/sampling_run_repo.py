"""Repository for SamplingRun table."""

from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.sampling_run import SamplingRun, SamplingRunStatus


class SamplingRunRepository(BaseRepository[SamplingRun]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(SamplingRun, session)

    async def get_by_status(self, status: SamplingRunStatus) -> Sequence[SamplingRun]:
        result = await self._exec(
            select(SamplingRun).where(SamplingRun.status == status).order_by(SamplingRun.created_at)
        )
        return result.all()

    async def get_by_source_job_id(self, source_job_id: int) -> Sequence[SamplingRun]:
        result = await self._exec(
            select(SamplingRun)
            .where(SamplingRun.source_job_id == source_job_id)
            .order_by(SamplingRun.created_at.desc())
        )
        return result.all()

    async def get_running(self) -> Optional[SamplingRun]:
        result = await self._exec(
            select(SamplingRun).where(SamplingRun.status == SamplingRunStatus.RUNNING).limit(1)
        )
        return result.first()

    async def update_status(
        self,
        sampling_run: SamplingRun,
        status: SamplingRunStatus,
        *,
        pid: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> SamplingRun:
        sampling_run.status = status
        sampling_run.updated_at = datetime.now(timezone.utc)
        if pid is not None:
            sampling_run.pid = pid
        if error_message is not None:
            sampling_run.error_message = error_message
        self._session.add(sampling_run)
        await self._session.flush()
        await self._session.refresh(sampling_run)
        return sampling_run

    async def update_output_path(self, sampling_run: SamplingRun, output_path: str) -> SamplingRun:
        sampling_run.output_path = output_path
        sampling_run.updated_at = datetime.now(timezone.utc)
        self._session.add(sampling_run)
        await self._session.flush()
        await self._session.refresh(sampling_run)
        return sampling_run

    async def update_log_path(self, sampling_run: SamplingRun, log_path: str) -> SamplingRun:
        sampling_run.log_path = log_path
        sampling_run.updated_at = datetime.now(timezone.utc)
        self._session.add(sampling_run)
        await self._session.flush()
        await self._session.refresh(sampling_run)
        return sampling_run

    async def update_progress_status(self, sampling_run: SamplingRun, status: Optional[str]) -> SamplingRun:
        sampling_run.progress_status = status
        if status is None:
            sampling_run.progress_step = None
            sampling_run.progress_total = None
        sampling_run.updated_at = datetime.now(timezone.utc)
        self._session.add(sampling_run)
        await self._session.flush()
        await self._session.refresh(sampling_run)
        return sampling_run

    async def update_progress(self, sampling_run: SamplingRun, step: int, total: int) -> SamplingRun:
        sampling_run.progress_step = step
        sampling_run.progress_total = total
        sampling_run.updated_at = datetime.now(timezone.utc)
        self._session.add(sampling_run)
        await self._session.flush()
        await self._session.refresh(sampling_run)
        return sampling_run

    async def clear_runtime_state(self, sampling_run: SamplingRun) -> SamplingRun:
        sampling_run.pid = None
        sampling_run.error_message = None
        await self.update_progress_status(sampling_run, None)
        return sampling_run

    async def clear_process_state(self, sampling_run: SamplingRun) -> SamplingRun:
        sampling_run.pid = None
        await self.update_progress_status(sampling_run, None)
        return sampling_run
