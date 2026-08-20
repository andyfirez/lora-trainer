"""Shared enqueue/cancel lifecycle for Runnable entities (Lora, Sampling)."""

from collections.abc import Awaitable, Callable

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.tables.runnable_mixin import RunnableMixin, RunnableStatus
from src.services.runnable import queue, runtime
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableNotCancellableError,
)

_TERMINAL_CANCEL_STATUSES = frozenset(
    {RunnableStatus.COMPLETED, RunnableStatus.FAILED, RunnableStatus.CANCELLED},
)


async def enqueue_runnable(
    session: AsyncSession,
    entity: RunnableMixin,
    *,
    kind: str,
    entity_id: int,
    before_enqueue: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Validate queue state, run optional domain hook, reset runtime, append to FIFO."""
    if entity.status in (RunnableStatus.QUEUED, RunnableStatus.RUNNING):
        raise RunnableAlreadyQueuedError(kind, entity_id)
    if before_enqueue is not None:
        await before_enqueue()
    runtime.clear_runtime(entity)
    await queue.enqueue(session, entity)


async def cancel_runnable(
    session: AsyncSession,
    entity: RunnableMixin,
    *,
    kind: str,
    entity_id: int,
    on_running: Callable[[], Awaitable[bool]] | None = None,
) -> RunnableMixin:
    """Cancel a queued or running entity.

    ``on_running`` is invoked when status is RUNNING. Return True to keep the entity
    running (e.g. defer cancel until checkpoint save). Otherwise the entity is
    cancelled; runtime fields are cleared only when it was not running.
    """
    if entity.status in _TERMINAL_CANCEL_STATUSES:
        raise RunnableNotCancellableError(kind, entity_id, entity.status)

    was_running = entity.status == RunnableStatus.RUNNING
    if was_running and on_running is not None:
        if await on_running():
            session.add(entity)
            await session.flush()
            return entity

    runtime.cancel(entity)
    if not was_running:
        runtime.clear_runtime(entity)
    session.add(entity)
    await session.flush()
    return entity
