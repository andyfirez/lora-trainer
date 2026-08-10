"""Repository for the Sampling table."""

from typing import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.sampling import Sampling


class SamplingRepository(BaseRepository[Sampling]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Sampling, session)

    async def list_all(self) -> Sequence[Sampling]:
        result = await self._exec(select(Sampling).order_by(Sampling.created_at.desc()))
        return result.all()
