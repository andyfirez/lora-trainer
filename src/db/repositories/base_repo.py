"""Generic async base repository over SQLModel."""

from collections.abc import Sequence
from typing import Any, Generic, Optional, TypeVar

from sqlalchemy import delete
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

ModelT = TypeVar("ModelT", bound=SQLModel)


class BaseRepository(Generic[ModelT]):
    def __init__(self, model: type[ModelT], session: AsyncSession) -> None:
        self._model = model
        self._session = session

    async def get_by_id(self, record_id: int) -> Optional[ModelT]:
        return await self._session.get(self._model, record_id)

    async def list_ordered(self, *order_by: Any) -> Sequence[ModelT]:
        statement = select(self._model)
        if order_by:
            statement = statement.order_by(*order_by)
        result = await self._exec(statement)
        return result.all()

    async def get_all(self) -> Sequence[ModelT]:
        return await self.list_ordered()

    async def get_by_field(self, field_name: str, value: Any, *, limit: int = 1) -> Optional[ModelT]:
        column = getattr(self._model, field_name)
        statement = select(self._model).where(column == value).limit(limit)
        result = await self._exec(statement)
        return result.first()

    async def add(self, record: ModelT) -> ModelT:
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return record

    def save(self, record: ModelT) -> ModelT:
        """Stage a record for flush without committing the transaction."""
        self._session.add(record)
        return record

    async def flush(self) -> None:
        await self._session.flush()

    async def refresh(self, record: ModelT) -> ModelT:
        await self._session.refresh(record)
        return record

    async def save_and_flush(self, record: ModelT) -> ModelT:
        self._session.add(record)
        await self._session.flush()
        return record

    async def save_flush_refresh(self, record: ModelT) -> ModelT:
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return record

    async def delete(self, record: ModelT) -> None:
        await self._session.delete(record)
        await self._session.flush()

    async def delete_where(self, **filters: Any) -> None:
        if not filters:
            raise ValueError("delete_where requires at least one filter")
        statement = delete(self._model)
        for field_name, value in filters.items():
            column = getattr(self._model, field_name)
            if isinstance(value, (list, tuple, set)):
                if not value:
                    raise ValueError(f"delete_where({field_name}=...) cannot be an empty collection")
                statement = statement.where(column.in_(value))
            else:
                statement = statement.where(column == value)
        await self._session.exec(statement)
        await self._session.flush()

    async def _exec(self, statement: Any) -> Any:
        return await self._session.exec(statement)
