"""FastAPI exception handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse

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
from src.services.tagging.exceptions import TaggingAlreadyRunningError


async def runnable_not_found_handler(request: Request, exc: RunnableNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def runnable_already_queued_handler(request: Request, exc: RunnableAlreadyQueuedError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def runnable_not_cancellable_handler(request: Request, exc: RunnableNotCancellableError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def runnable_not_resumable_handler(request: Request, exc: RunnableNotResumableError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def runnable_operation_not_supported_handler(
    request: Request,
    exc: RunnableOperationNotSupportedError,
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def runnable_validation_handler(request: Request, exc: RunnableValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.message})


async def lora_not_found_handler(request: Request, exc: LoraNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def lora_name_conflict_handler(request: Request, exc: LoraNameConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def lora_reproduce_handler(request: Request, exc: LoraReproduceError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def lora_checkpoint_not_found_handler(request: Request, exc: LoraCheckpointNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def sampling_lora_path_not_found_handler(request: Request, exc: SamplingLoRAPathNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


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


async def tagging_already_running_handler(request: Request, exc: TaggingAlreadyRunningError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})
