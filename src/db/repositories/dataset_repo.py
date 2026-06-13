"""Repository for Dataset table."""

from typing import Optional

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.dataset import Dataset


class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Dataset, session)

    async def get_by_name(self, name: str) -> Optional[Dataset]:
        result = await self._exec(
            select(Dataset).where(Dataset.name == name).limit(1)
        )
        return result.first()
