"""Repository for QueueEntry table."""

from typing import Optional, Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.queue_entry import QueueEntry


class QueueRepository(BaseRepository[QueueEntry]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(QueueEntry, session)

    async def get_ordered(self) -> Sequence[QueueEntry]:
        result = await self._exec(select(QueueEntry).order_by(QueueEntry.position))
        return result.all()

    async def get_by_job_id(self, job_id: int) -> Optional[QueueEntry]:
        result = await self._exec(
            select(QueueEntry).where(QueueEntry.job_id == job_id).limit(1)
        )
        return result.first()

    async def get_next(self) -> Optional[QueueEntry]:
        result = await self._exec(
            select(QueueEntry).order_by(QueueEntry.position).limit(1)
        )
        return result.first()

    async def get_max_position(self) -> int:
        result = await self._exec(select(QueueEntry.position).order_by(QueueEntry.position.desc()).limit(1))
        value = result.first()
        return value if value is not None else 0

    async def shift_positions_down(self, after_position: int) -> None:
        result = await self._exec(
            select(QueueEntry).where(QueueEntry.position > after_position)
        )
        entries = result.all()
        for entry in entries:
            entry.position -= 1
            self._session.add(entry)
        await self._session.flush()
