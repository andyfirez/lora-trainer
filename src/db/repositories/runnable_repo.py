"""Repository helpers for Runnable entities (Lora, Sampling)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from src.db.tables.runnable_mixin import RunnableMixin
from src.services.runnable.lifecycle import cancel_runnable, enqueue_runnable

RunnableT = TypeVar("RunnableT", bound=RunnableMixin)


class RunnableRepositoryMixin(Generic[RunnableT]):
    _session: object

    async def enqueue_runnable(
        self,
        entity: RunnableT,
        *,
        kind: str,
        entity_id: int,
        before_enqueue: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        await enqueue_runnable(
            self._session,  # type: ignore[arg-type]
            entity,
            kind=kind,
            entity_id=entity_id,
            before_enqueue=before_enqueue,
        )

    async def cancel_runnable(
        self,
        entity: RunnableT,
        *,
        kind: str,
        entity_id: int,
        on_running: Callable[[], Awaitable[bool]] | None = None,
    ) -> RunnableT:
        return await cancel_runnable(
            self._session,  # type: ignore[arg-type]
            entity,
            kind=kind,
            entity_id=entity_id,
            on_running=on_running,
        )
