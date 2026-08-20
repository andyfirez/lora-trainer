"""Handler interface for dispatching a Runnable entity to its subprocess runner."""

from abc import ABC, abstractmethod

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.tables.runnable_mixin import RunnableMixin, RunnableStatus
from src.services.runnable import runtime


class RunnableHandler(ABC):
    @abstractmethod
    def build_command(self, entity_id: int) -> list[str]:
        """Return subprocess argv for the given entity id."""

    @abstractmethod
    async def _load_entity(self, session: AsyncSession, entity_id: int) -> RunnableMixin | None:
        """Load the entity for finalize; return None when missing."""

    async def finalize(
        self,
        session: AsyncSession,
        entity_id: int,
        exit_code: int,
        *,
        error_message: str | None = None,
    ) -> None:
        """Apply the terminal status and any domain-specific completion logic.

        Called by the worker after the subprocess exits. `error_message` is a
        pre-formatted failure summary the worker derived from subprocess output,
        used when `exit_code != 0` and the entity has no error message yet.
        Does not commit — the caller is responsible for the transaction.
        """
        entity = await self._load_entity(session, entity_id)
        if entity is None or entity.status != RunnableStatus.RUNNING:
            return
        await self._finalize_running_entity(session, entity, exit_code, error_message=error_message)

    async def _finalize_running_entity(
        self,
        session: AsyncSession,
        entity: RunnableMixin,
        exit_code: int,
        *,
        error_message: str | None = None,
    ) -> None:
        final_status = RunnableStatus.COMPLETED if exit_code == 0 else RunnableStatus.FAILED
        self._before_mark_finished(entity, final_status)
        runtime.mark_finished(
            entity,
            final_status,
            error_message=(entity.error_message or error_message) if final_status == RunnableStatus.FAILED else None,
        )
        session.add(entity)
        await session.flush()
        if final_status == RunnableStatus.COMPLETED:
            await self.on_completed(session, entity)

    def _before_mark_finished(self, entity: RunnableMixin, final_status: RunnableStatus) -> None:
        """Hook for domain-specific tweaks immediately before mark_finished."""

    async def on_completed(self, session: AsyncSession, entity: RunnableMixin) -> None:
        """Hook invoked after a successful run is persisted."""
