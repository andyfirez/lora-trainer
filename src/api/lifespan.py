"""Application startup/shutdown lifecycle."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import AsyncContextManager

from fastapi import FastAPI

from src.db.session import run_migrations
from src.services.worker.service import RunnableWorker

logger = logging.getLogger(__name__)


def create_lifespan(
    *,
    worker_factory: Callable[[], RunnableWorker] | None = None,
    migrate: Callable[[], Awaitable[None]] | None = None,
) -> Callable[[FastAPI], AsyncContextManager[None]]:
    """Build a FastAPI lifespan that runs migrations and the runnable worker."""

    create_worker = worker_factory or (lambda: RunnableWorker(echo_subprocess_output=False))
    run_migrate = migrate or run_migrations

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting up — running database migrations")
        await run_migrate()
        worker = create_worker()
        await worker.start()
        app.state.runnable_worker = worker
        try:
            yield
        finally:
            await worker.stop()
            logger.info("Shutting down")

    return lifespan
