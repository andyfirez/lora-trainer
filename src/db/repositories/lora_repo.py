"""Repository for the Lora table."""

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.lora import Lora


class LoraRepository(BaseRepository[Lora]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Lora, session)

    async def get_by_name(self, name: str) -> Lora | None:
        return await self.get_by_field("name", name)

    async def get_by_relative_path(self, relative_path: str) -> Lora | None:
        return await self.get_by_field("relative_path", relative_path)
