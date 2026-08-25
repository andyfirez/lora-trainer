"""Repository for the Sampling table."""

from collections.abc import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.repositories.runnable_repo import RunnableRepositoryMixin
from src.db.tables.sampling import Sampling, SamplingKind


class SamplingRepository(RunnableRepositoryMixin[Sampling], BaseRepository[Sampling]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Sampling, session)

    async def list_by_kind(self, kind: SamplingKind) -> Sequence[Sampling]:
        statement = select(Sampling).where(Sampling.kind == kind).order_by(Sampling.created_at.desc())
        result = await self._exec(statement)
        return result.all()
