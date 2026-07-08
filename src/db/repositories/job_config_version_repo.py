"""Repository for JobConfigVersion table."""

from typing import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.job_config_version import JobConfigVersion


class JobConfigVersionRepository(BaseRepository[JobConfigVersion]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(JobConfigVersion, session)

    async def get_by_config_id(self, config_id: int) -> Sequence[JobConfigVersion]:
        result = await self._exec(
            select(JobConfigVersion)
            .where(JobConfigVersion.config_id == config_id)
            .order_by(JobConfigVersion.version)
        )
        return result.all()

    async def get_by_config_and_version(
        self,
        config_id: int,
        version: int,
    ) -> JobConfigVersion | None:
        result = await self._exec(
            select(JobConfigVersion)
            .where(JobConfigVersion.config_id == config_id)
            .where(JobConfigVersion.version == version)
        )
        return result.first()
