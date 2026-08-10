"""Handler interface for dispatching a Runnable entity to its subprocess runner."""

from abc import ABC, abstractmethod

from sqlmodel.ext.asyncio.session import AsyncSession


class RunnableHandler(ABC):
    @abstractmethod
    def build_command(self, entity_id: int) -> list[str]:
        """Return subprocess argv for the given entity id."""

    @abstractmethod
    def validate_config_yaml(self, config_yaml: str) -> None:
        """Validate config YAML before enqueueing."""

    @abstractmethod
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
