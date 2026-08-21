"""API tests for the generic AppError exception handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.api.exception_handlers import register_exception_handlers
from src.api.lifespan import create_lifespan
from src.services.common.exceptions import AppError, NameConflictError, NotFoundError
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetImageNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetNotPreparedError,
    DatasetPreprocessError,
    DatasetResolutionMismatchError,
    DatasetTargetResolutionNotSetError,
    InvalidDatasetFilenameError,
)
from src.services.loras.exceptions import (
    LoraCheckpointNotFoundError,
    LoraNameConflictError,
    LoraNotFoundError,
    LoraReproduceError,
)
from src.services.png_info.exceptions import InvalidImageError
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableNotCancellableError,
    RunnableNotFoundError,
    RunnableNotResumableError,
    RunnableOperationNotSupportedError,
    RunnableValidationError,
)
from src.services.sampling.exceptions import (
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)
from src.services.settings.exceptions import (
    EmptySettingsPatchError,
    InvalidGpuDefaultsError,
)
from src.services.tagging.exceptions import TaggingAlreadyRunningError

STATUS_CASES: list[tuple[AppError, int]] = [
    (RunnableNotFoundError("LoRA", 1), 404),
    (RunnableAlreadyQueuedError("LoRA", 1), 409),
    (RunnableNotCancellableError("LoRA", 1, "completed"), 409),
    (RunnableNotResumableError("LoRA", 1, "completed"), 409),
    (RunnableOperationNotSupportedError("LoRA", 1, "resume"), 400),
    (RunnableValidationError("bad yaml"), 422),
    (LoraNotFoundError(1), 404),
    (LoraNameConflictError("demo"), 409),
    (LoraReproduceError(1), 422),
    (LoraCheckpointNotFoundError(1), 404),
    (SamplingLoRAPathNotFoundError("/missing.safetensors"), 404),
    (SamplingPromptsNotConfiguredError(), 422),
    (DatasetNotFoundError(1), 404),
    (DatasetNameConflictError("demo"), 409),
    (DatasetDirectoryNotFoundError("/missing"), 404),
    (DatasetImageNotFoundError("cat.png"), 404),
    (InvalidDatasetFilenameError("../x.png"), 400),
    (DatasetNotPreparedError(1, "demo", "no crops"), 422),
    (DatasetResolutionMismatchError(1, "demo", 512, 1024), 422),
    (DatasetTargetResolutionNotSetError(1), 422),
    (DatasetPreprocessError("failed"), 422),
    (TaggingAlreadyRunningError(1), 409),
    (InvalidImageError(), 422),
    (EmptySettingsPatchError(), 422),
    (InvalidGpuDefaultsError("bad gpu"), 422),
]


def _app_for(exc: AppError) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    return app


@pytest.mark.parametrize(
    ("exc", "base"),
    [
        (LoraNotFoundError(1), NotFoundError),
        (DatasetNotFoundError(1), NotFoundError),
        (RunnableNotFoundError("Sampling", 1), NotFoundError),
        (LoraNameConflictError("demo"), NameConflictError),
        (DatasetNameConflictError("demo"), NameConflictError),
    ],
)
def test_resource_errors_inherit_shared_bases(exc: AppError, base: type[AppError]) -> None:
    assert isinstance(exc, base)


@pytest.mark.asyncio
@pytest.mark.parametrize(("exc", "status_code"), STATUS_CASES)
async def test_app_error_handler_maps_status_and_detail(exc: AppError, status_code: int) -> None:
    transport = ASGITransport(app=_app_for(exc))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")
    assert response.status_code == status_code
    assert response.json() == {"detail": str(exc)}


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
