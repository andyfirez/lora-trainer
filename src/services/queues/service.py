"""Business logic for the training queue."""

from typing import Sequence

from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.tables.job import Job
from src.db.tables.queue_entry import QueueEntry
from src.services.queues.exceptions import QueueEntryNotFoundError


class QueuesService:
    def __init__(
        self,
        queue_repo: QueueRepository,
        job_repo: JobRepository,
    ) -> None:
        self._queue_repo = queue_repo
        self._job_repo = job_repo

    async def list_queue(self) -> Sequence[QueueEntry]:
        return await self._queue_repo.get_ordered()

    async def list_queue_with_jobs(self) -> Sequence[tuple[QueueEntry, Job]]:
        entries = await self._queue_repo.get_ordered()
        result: list[tuple[QueueEntry, Job]] = []
        for entry in entries:
            job = await self._job_repo.get_by_id(entry.job_id)
            if job is not None:
                result.append((entry, job))
        return result

    async def move_to_top(self, job_id: int) -> QueueEntry:
        entry = await self._queue_repo.get_by_job_id(job_id)
        if entry is None:
            raise QueueEntryNotFoundError(job_id)
        entry.position = 0
        self._queue_repo._session.add(entry)
        await self._queue_repo._session.flush()
        await self._queue_repo.shift_positions_down(0)
        entry.position = 1
        self._queue_repo._session.add(entry)
        await self._queue_repo._session.flush()
        await self._queue_repo._session.refresh(entry)
        return entry

    async def get_next_pending(self) -> QueueEntry | None:
        return await self._queue_repo.get_next()
