"""Shared DB update helpers for runnable subprocess runners."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from src.db.session import session_factory
from src.db.tables.runnable_mixin import RunnableStatus

T = TypeVar("T")


async def get_runnable_entity(repo_cls: type, entity_id: int) -> T | None:
    async with session_factory() as session:
        repo = repo_cls(session)
        return await repo.get_by_id(entity_id)


async def get_active_runnable(repo_cls: type, entity_id: int) -> T | None:
    entity = await get_runnable_entity(repo_cls, entity_id)
    if entity is None or entity.status == RunnableStatus.CANCELLED:
        return None
    return entity


async def update_runnable_entity(
    repo_cls: type,
    entity_id: int,
    mutator: Callable[[T], None],
    *,
    skip_if_cancelled: bool = False,
) -> T | None:
    async with session_factory() as session:
        repo = repo_cls(session)
        entity = await repo.get_by_id(entity_id)
        if entity is None:
            return None
        if skip_if_cancelled and entity.status == RunnableStatus.CANCELLED:
            return None
        mutator(entity)
        session.add(entity)
        await session.commit()
        return entity
