"""Generic async base repository over SQLModel."""

from typing import Any, Generic, Optional, Sequence, TypeVar

from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

ModelT = TypeVar("ModelT", bound=SQLModel)


class BaseRepository(Generic[ModelT]):
    def __init__(self, model: type[ModelT], session: AsyncSession) -> None:
        self._model = model
        self._session = session

    async def get_by_id(self, record_id: int) -> Optional[ModelT]:
        return await self._session.get(self._model, record_id)

    async def get_all(self) -> Sequence[ModelT]:
        result = await self._session.exec(select(self._model))
        return result.all()

    async def add(self, record: ModelT) -> ModelT:
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return record

    async def delete(self, record: ModelT) -> None:
        await self._session.delete(record)
        await self._session.flush()

    async def _exec(self, statement: Any) -> Any:
        return await self._session.exec(statement)
