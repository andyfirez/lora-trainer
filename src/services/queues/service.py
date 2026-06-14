"""Business logic for the training queue."""

from typing import Sequence

from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.sampling_run_repo import SamplingRunRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.queue_entry import QueueEntry, QueueItemType
from src.db.tables.sampling_run import SamplingRun
from src.db.tables.training_job import TrainingJob
from src.services.queues.exceptions import QueueEntryNotFoundError


class QueuesService:
    def __init__(
        self,
        queue_repo: QueueRepository,
        job_repo: TrainingJobRepository,
        sampling_run_repo: SamplingRunRepository,
    ) -> None:
        self._queue_repo = queue_repo
        self._job_repo = job_repo
        self._sampling_run_repo = sampling_run_repo

    async def list_queue(self) -> Sequence[QueueEntry]:
        return await self._queue_repo.get_ordered()

    async def list_queue_with_items(self) -> Sequence[tuple[QueueEntry, TrainingJob | SamplingRun]]:
        entries = await self._queue_repo.get_ordered()
        result: list[tuple[QueueEntry, TrainingJob | SamplingRun]] = []
        for entry in entries:
            if entry.item_type == QueueItemType.TRAINING:
                job = await self._job_repo.get_by_id(entry.item_id)
                if job is not None:
                    result.append((entry, job))
            elif entry.item_type == QueueItemType.SAMPLING:
                sampling_run = await self._sampling_run_repo.get_by_id(entry.item_id)
                if sampling_run is not None:
                    result.append((entry, sampling_run))
        return result

    async def move_to_top(self, item_type: QueueItemType, item_id: int) -> QueueEntry:
        entry = await self._queue_repo.get_by_item(item_type, item_id)
        if entry is None:
            raise QueueEntryNotFoundError(item_id)
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
