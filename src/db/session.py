"""SQLModel + aiosqlite engine, session factory, and unit of work."""

from contextlib import asynccontextmanager
from types import TracebackType
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.alembic_runner import run_migrations as _run_alembic_migrations
from src.settings.app_settings import settings


def register_all_tables() -> None:
    """Import every table model so SQLAlchemy metadata resolves foreign keys."""
    import src.db.tables.dataset  # noqa: F401
    import src.db.tables.job  # noqa: F401
    import src.db.tables.job_config  # noqa: F401
    import src.db.tables.queue_entry  # noqa: F401


register_all_tables()


def _create_engine() -> AsyncEngine:
    db_path = settings.database.path
    url = f"sqlite+aiosqlite:///{db_path}"
    return create_async_engine(
        url,
        echo=settings.database.echo,
        connect_args={"timeout": 30},
    )


engine: AsyncEngine = _create_engine()

session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class UnitOfWork:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self.session: AsyncSession | None = None

    async def __aenter__(self) -> "UnitOfWork":
        self.session = self._factory()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.session is not None:
            if exc_type is not None:
                await self.session.rollback()
            await self.session.close()

    async def commit(self) -> None:
        if self.session is not None:
            await self.session.commit()

    async def rollback(self) -> None:
        if self.session is not None:
            await self.session.rollback()


@asynccontextmanager
async def get_uow() -> AsyncGenerator[UnitOfWork, None]:
    async with UnitOfWork(session_factory) as uow:
        yield uow


async def run_migrations() -> None:
    """Apply Alembic migrations (first-run bootstrap and schema upgrades)."""
    register_all_tables()
    _run_alembic_migrations()
