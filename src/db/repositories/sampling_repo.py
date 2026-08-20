"""Repository for the Sampling table."""

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.repositories.runnable_repo import RunnableRepositoryMixin
from src.db.tables.sampling import Sampling


class SamplingRepository(RunnableRepositoryMixin[Sampling], BaseRepository[Sampling]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Sampling, session)
