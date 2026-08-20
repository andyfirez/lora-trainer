"""Repository for Dataset table."""

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.dataset import Dataset


class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Dataset, session)

    async def get_by_name(self, name: str) -> Dataset | None:
        return await self.get_by_field("name", name)
