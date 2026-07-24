"""Repository for TrainedLora table."""

from typing import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.trained_lora import TrainedLora


class TrainedLoraRepository(BaseRepository[TrainedLora]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(TrainedLora, session)

    async def get_by_job_id(self, job_id: int) -> TrainedLora | None:
        result = await self._exec(select(TrainedLora).where(TrainedLora.job_id == job_id))
        return result.first()

    async def get_by_name(self, name: str) -> TrainedLora | None:
        result = await self._exec(select(TrainedLora).where(TrainedLora.name == name))
        return result.first()

    async def get_by_relative_path(self, relative_path: str) -> TrainedLora | None:
        result = await self._exec(
            select(TrainedLora).where(TrainedLora.relative_path == relative_path)
        )
        return result.first()

    async def list_all(self) -> Sequence[TrainedLora]:
        result = await self._exec(select(TrainedLora).order_by(TrainedLora.created_at.desc()))
        return result.all()
