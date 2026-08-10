"""Repository for the Lora table."""

from typing import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.lora import Lora


class LoraRepository(BaseRepository[Lora]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Lora, session)

    async def get_by_name(self, name: str) -> Lora | None:
        result = await self._exec(select(Lora).where(Lora.name == name))
        return result.first()

    async def get_by_relative_path(self, relative_path: str) -> Lora | None:
        result = await self._exec(select(Lora).where(Lora.relative_path == relative_path))
        return result.first()

    async def list_all(self) -> Sequence[Lora]:
        result = await self._exec(select(Lora).order_by(Lora.created_at.desc()))
        return result.all()
