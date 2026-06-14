"""Repository for JobConfig table."""

from typing import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.job_config import ConfigType, JobConfig


class JobConfigRepository(BaseRepository[JobConfig]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(JobConfig, session)

    async def get_by_type(self, config_type: ConfigType) -> Sequence[JobConfig]:
        result = await self._exec(
            select(JobConfig)
            .where(JobConfig.config_type == config_type)
            .order_by(JobConfig.created_at)
        )
        return result.all()
