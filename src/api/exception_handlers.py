"""FastAPI exception handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse

from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
)
from src.services.jobs.exceptions import (
    JobCheckpointNotFoundError,
    JobAlreadyQueuedError,
    JobNotCancellableError,
    JobNotFoundError,
    JobNotResumableError,
)
from src.services.queues.exceptions import QueueEntryNotFoundError


async def job_not_found_handler(request: Request, exc: JobNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def job_already_queued_handler(request: Request, exc: JobAlreadyQueuedError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def job_not_cancellable_handler(request: Request, exc: JobNotCancellableError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def job_not_resumable_handler(request: Request, exc: JobNotResumableError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def job_checkpoint_not_found_handler(request: Request, exc: JobCheckpointNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def queue_entry_not_found_handler(request: Request, exc: QueueEntryNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def dataset_not_found_handler(request: Request, exc: DatasetNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def dataset_name_conflict_handler(request: Request, exc: DatasetNameConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def dataset_dir_not_found_handler(request: Request, exc: DatasetDirectoryNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
