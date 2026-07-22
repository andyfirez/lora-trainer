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
    job_already_queued_handler,
    job_checkpoint_not_found_handler,
    job_config_not_found_handler,
    job_config_validation_handler,
    job_not_cancellable_handler,
    job_not_found_handler,
    job_not_resumable_handler,
    job_operation_not_supported_handler,
    queue_entry_not_found_handler,
    sampling_checkpoints_not_found_handler,
    sampling_lora_path_not_found_handler,
    sampling_prompts_not_configured_handler,
    trained_lora_not_found_handler,
)
from src.api.routers import configs, datasets, files, jobs, loras, queues
from src.db.session import run_migrations
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
from src.services.queues.exceptions import QueueEntryNotFoundError
from src.services.sampling.exceptions import (
    SamplingCheckpointsNotFoundError,
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)
from src.services.worker.service import QueueWorker
from src.settings.app_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_EXCEPTION_HANDLERS: dict[type[Exception], object] = {
    JobNotFoundError: job_not_found_handler,
    JobAlreadyQueuedError: job_already_queued_handler,
    JobNotCancellableError: job_not_cancellable_handler,
    JobNotResumableError: job_not_resumable_handler,
    JobCheckpointNotFoundError: job_checkpoint_not_found_handler,
    JobOperationNotSupportedError: job_operation_not_supported_handler,
    QueueEntryNotFoundError: queue_entry_not_found_handler,
    JobConfigNotFoundError: job_config_not_found_handler,
    JobConfigValidationError: job_config_validation_handler,
    TrainedLoraNotFoundError: trained_lora_not_found_handler,
    SamplingLoRAPathNotFoundError: sampling_lora_path_not_found_handler,
    SamplingCheckpointsNotFoundError: sampling_checkpoints_not_found_handler,
    SamplingPromptsNotConfiguredError: sampling_prompts_not_configured_handler,
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
    worker = QueueWorker(echo_subprocess_output=False)
    await worker.start()
    app.state.queue_worker = worker
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

app.include_router(configs.router)
app.include_router(loras.router)
app.include_router(jobs.router)
app.include_router(queues.router)
app.include_router(datasets.router)
app.include_router(files.router)


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
