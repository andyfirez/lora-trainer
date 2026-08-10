"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exception_handlers import (
    dataset_dir_not_found_handler,
    dataset_image_not_found_handler,
    dataset_name_conflict_handler,
    dataset_not_found_handler,
    dataset_not_prepared_handler,
    dataset_preprocess_handler,
    dataset_resolution_mismatch_handler,
    dataset_target_resolution_not_set_handler,
    invalid_dataset_filename_handler,
    lora_checkpoint_not_found_handler,
    lora_name_conflict_handler,
    lora_not_found_handler,
    lora_reproduce_handler,
    runnable_already_queued_handler,
    runnable_not_cancellable_handler,
    runnable_not_found_handler,
    runnable_not_resumable_handler,
    runnable_operation_not_supported_handler,
    runnable_validation_handler,
    sampling_lora_path_not_found_handler,
    sampling_prompts_not_configured_handler,
    tagging_already_running_handler,
)
from src.api.routers import (
    datasets,
    files,
    loras,
    png_info,
    samplings,
    storage,
)
from src.api.routers import (
    settings as settings_router,
)
from src.db.session import run_migrations
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
from src.services.worker.service import RunnableWorker
from src.settings.app_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_EXCEPTION_HANDLERS: dict[type[Exception], object] = {
    RunnableNotFoundError: runnable_not_found_handler,
    RunnableAlreadyQueuedError: runnable_already_queued_handler,
    RunnableNotCancellableError: runnable_not_cancellable_handler,
    RunnableNotResumableError: runnable_not_resumable_handler,
    RunnableOperationNotSupportedError: runnable_operation_not_supported_handler,
    RunnableValidationError: runnable_validation_handler,
    LoraNotFoundError: lora_not_found_handler,
    LoraNameConflictError: lora_name_conflict_handler,
    LoraReproduceError: lora_reproduce_handler,
    LoraCheckpointNotFoundError: lora_checkpoint_not_found_handler,
    SamplingLoRAPathNotFoundError: sampling_lora_path_not_found_handler,
    SamplingPromptsNotConfiguredError: sampling_prompts_not_configured_handler,
    TaggingAlreadyRunningError: tagging_already_running_handler,
    DatasetNotFoundError: dataset_not_found_handler,
    DatasetNameConflictError: dataset_name_conflict_handler,
    DatasetDirectoryNotFoundError: dataset_dir_not_found_handler,
    DatasetImageNotFoundError: dataset_image_not_found_handler,
    InvalidDatasetFilenameError: invalid_dataset_filename_handler,
    DatasetNotPreparedError: dataset_not_prepared_handler,
    DatasetResolutionMismatchError: dataset_resolution_mismatch_handler,
    DatasetTargetResolutionNotSetError: dataset_target_resolution_not_set_handler,
    DatasetPreprocessError: dataset_preprocess_handler,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up — running database migrations")
    await run_migrations()
    worker = RunnableWorker(echo_subprocess_output=False)
    await worker.start()
    app.state.runnable_worker = worker
    yield
    await worker.stop()
    logger.info("Shutting down")


app = FastAPI(title="LoRA Trainer API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for exc_type, handler in _EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exc_type, handler)  # type: ignore[arg-type]

app.include_router(loras.router)
app.include_router(samplings.router)
app.include_router(datasets.router)
app.include_router(files.router)
app.include_router(settings_router.router)
app.include_router(storage.router)
app.include_router(png_info.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run(
        "src.api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
