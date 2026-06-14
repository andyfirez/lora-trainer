"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exception_handlers import (
    dataset_dir_not_found_handler,
    dataset_name_conflict_handler,
    dataset_not_found_handler,
    job_already_queued_handler,
    job_checkpoint_not_found_handler,
    job_not_cancellable_handler,
    job_not_found_handler,
    job_not_resumable_handler,
    queue_entry_not_found_handler,
    sampling_checkpoints_not_found_handler,
    sampling_lora_path_not_found_handler,
    sampling_prompts_not_configured_handler,
    sampling_run_already_queued_handler,
    sampling_run_not_cancellable_handler,
    sampling_run_not_found_handler,
)
from src.api.routers import datasets, files, jobs, queues, sampling_runs
from src.db.session import create_tables
from src.services.worker.service import QueueWorker
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
)
from src.services.jobs.exceptions import (
    JobAlreadyQueuedError,
    JobCheckpointNotFoundError,
    JobNotCancellableError,
    JobNotFoundError,
    JobNotResumableError,
)
from src.services.queues.exceptions import QueueEntryNotFoundError
from src.services.sampling.exceptions import (
    SamplingCheckpointsNotFoundError,
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
    SamplingRunAlreadyQueuedError,
    SamplingRunNotCancellableError,
    SamplingRunNotFoundError,
)
from src.settings.app_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up — creating database tables")
    await create_tables()
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

app.add_exception_handler(JobNotFoundError, job_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(JobAlreadyQueuedError, job_already_queued_handler)  # type: ignore[arg-type]
app.add_exception_handler(JobNotCancellableError, job_not_cancellable_handler)  # type: ignore[arg-type]
app.add_exception_handler(JobNotResumableError, job_not_resumable_handler)  # type: ignore[arg-type]
app.add_exception_handler(JobCheckpointNotFoundError, job_checkpoint_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(QueueEntryNotFoundError, queue_entry_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(SamplingRunNotFoundError, sampling_run_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(SamplingRunAlreadyQueuedError, sampling_run_already_queued_handler)  # type: ignore[arg-type]
app.add_exception_handler(SamplingRunNotCancellableError, sampling_run_not_cancellable_handler)  # type: ignore[arg-type]
app.add_exception_handler(SamplingLoRAPathNotFoundError, sampling_lora_path_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(SamplingCheckpointsNotFoundError, sampling_checkpoints_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(SamplingPromptsNotConfiguredError, sampling_prompts_not_configured_handler)  # type: ignore[arg-type]
app.add_exception_handler(DatasetNotFoundError, dataset_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(DatasetNameConflictError, dataset_name_conflict_handler)  # type: ignore[arg-type]
app.add_exception_handler(DatasetDirectoryNotFoundError, dataset_dir_not_found_handler)  # type: ignore[arg-type]

app.include_router(jobs.router)
app.include_router(queues.router)
app.include_router(sampling_runs.router)
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
