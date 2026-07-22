"""FastAPI exception handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse

from src.services.configs.exceptions import (
    JobConfigNotFoundError,
    JobConfigValidationError,
)
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
from src.services.jobs.exceptions import (
    JobAlreadyQueuedError,
    JobCheckpointNotFoundError,
    JobNotCancellableError,
    JobNotFoundError,
    JobNotResumableError,
    JobOperationNotSupportedError,
)
from src.services.loras.exceptions import TrainedLoraNotFoundError
from src.services.sampling.exceptions import (
    SamplingCheckpointsNotFoundError,
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)


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


async def job_operation_not_supported_handler(
    request: Request,
    exc: JobOperationNotSupportedError,
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def job_config_not_found_handler(request: Request, exc: JobConfigNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def job_config_validation_handler(request: Request, exc: JobConfigValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.message})


async def trained_lora_not_found_handler(request: Request, exc: TrainedLoraNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def sampling_lora_path_not_found_handler(request: Request, exc: SamplingLoRAPathNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def sampling_checkpoints_not_found_handler(
    request: Request,
    exc: SamplingCheckpointsNotFoundError,
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def sampling_prompts_not_configured_handler(
    request: Request,
    exc: SamplingPromptsNotConfiguredError,
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def dataset_not_found_handler(request: Request, exc: DatasetNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def dataset_name_conflict_handler(request: Request, exc: DatasetNameConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def dataset_dir_not_found_handler(request: Request, exc: DatasetDirectoryNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def dataset_image_not_found_handler(request: Request, exc: DatasetImageNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def invalid_dataset_filename_handler(request: Request, exc: InvalidDatasetFilenameError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def dataset_not_prepared_handler(request: Request, exc: DatasetNotPreparedError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def dataset_resolution_mismatch_handler(
    request: Request,
    exc: DatasetResolutionMismatchError,
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def dataset_target_resolution_not_set_handler(
    request: Request,
    exc: DatasetTargetResolutionNotSetError,
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def dataset_preprocess_handler(request: Request, exc: DatasetPreprocessError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
