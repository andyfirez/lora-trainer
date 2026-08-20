"""API tests for the generic AppError exception handler."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.api.exception_handlers import register_exception_handlers
from src.services.common.exceptions import AppError
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


@pytest.mark.parametrize(("exc", "status_code"), STATUS_CASES)
def test_domain_errors_declare_http_status(exc: AppError, status_code: int) -> None:
    assert isinstance(exc, AppError)
    assert exc.status_code == status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(("exc", "status_code"), STATUS_CASES)
async def test_app_error_handler_maps_status_and_detail(exc: AppError, status_code: int) -> None:
    transport = ASGITransport(app=_app_for(exc))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")
    assert response.status_code == status_code
    assert response.json() == {"detail": str(exc)}
