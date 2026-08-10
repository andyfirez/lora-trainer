"""Shared global FIFO queue for Runnable entities (Lora, Sampling)."""

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.runnable_queries import (
    RunnableKind,
    max_queue_position,
    next_queued,
    running_any,
)
from src.db.tables.runnable_mixin import RunnableMixin, RunnableStatus

__all__ = ["enqueue", "next_queued", "running_any", "RunnableKind"]


async def enqueue(session: AsyncSession, entity: RunnableMixin) -> None:
    """Append entity to the back of the global FIFO queue."""
    position = await max_queue_position(session) + 1
    entity.status = RunnableStatus.QUEUED
    entity.queue_position = position
    entity.error_message = None
    session.add(entity)
    await session.flush()
    await session.refresh(entity)
