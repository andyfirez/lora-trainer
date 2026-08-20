from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from src.api.lifespan import create_lifespan


@pytest.mark.asyncio
async def test_create_lifespan_starts_and_stops_worker() -> None:
    worker = MagicMock()
    worker.start = AsyncMock()
    worker.stop = AsyncMock()
    migrate = AsyncMock()
    app = FastAPI()

    lifespan = create_lifespan(worker_factory=lambda: worker, migrate=migrate)
    async with lifespan(app):
        migrate.assert_awaited_once()
        worker.start.assert_awaited_once()
        assert app.state.runnable_worker is worker

    worker.stop.assert_awaited_once()
