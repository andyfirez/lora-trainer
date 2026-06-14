"""Training runner — CLI entry point spawned by the queue worker.

Usage:
    python -m src.trainer.runner --job-id <id>

Loads the job's config_yaml from SQLite, runs the SDXL LoRA trainer,
and writes progress + final status back to the DB synchronously.
"""

import argparse
import asyncio
import logging
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.session import session_factory
from src.db.tables.training_job import JobStatus, TrainingJob
from src.settings.app_settings import settings
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import MetricLogger, build_loss_log_path, reset_loss_log
from src.trainer.sdxl.trainer import SDXLLoRATrainer
from src.trainer.training_log import JobTrainingLogger, setup_tensorboard_writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Background event loop for non-blocking DB progress updates from the training thread.
_progress_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
threading.Thread(
    target=_progress_loop.run_forever,
    daemon=True,
    name="progress-db-loop",
).start()


async def _get_active_job(repo: TrainingJobRepository, job_id: int) -> TrainingJob | None:
    job = await repo.get_by_id(job_id)
    if job is None or job.status == JobStatus.CANCELLED:
        return None
    return job


async def _update_progress(
    job_id: int,
    step: int,
    total: int,
    loss: float,
    avr_loss: float,
    epoch: int,
    epoch_total: int,
) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await _get_active_job(repo, job_id)
        if job is not None:
            await repo.update_progress(
                job,
                step,
                total,
                loss=loss,
                avr_loss=avr_loss,
                epoch=epoch,
                epoch_total=epoch_total,
            )
            await session.commit()


async def _update_sampling_status(job_id: int, status: str | None) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await _get_active_job(repo, job_id)
        if job is not None:
            await repo.update_sampling_status(job, status)
            await session.commit()


async def _update_sampling_progress(job_id: int, step: int, total: int) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await _get_active_job(repo, job_id)
        if job is not None:
            await repo.update_sampling_progress(job, step, total)
            await session.commit()


async def _update_status(job_id: int, status: JobStatus, error: str | None = None) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            if job.status == JobStatus.CANCELLED and status != JobStatus.CANCELLED:
                return
            await repo.update_status(job, status, error_message=error)
            await session.commit()


async def _set_log_path(job_id: int, log_path: str) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            await repo.update_log_path(job, log_path)
            await session.commit()


def _submit_to_progress_loop(coro: Coroutine[Any, Any, None]) -> None:
    future = asyncio.run_coroutine_threadsafe(coro, _progress_loop)

    def _log_exception(fut: asyncio.Future[object]) -> None:
        try:
            fut.result()
        except Exception:
            logger.exception("Progress DB update failed")

    future.add_done_callback(_log_exception)


def _make_progress_callback(job_id: int):
    def callback(
        step: int,
        total: int,
        loss: float,
        avr_loss: float,
        epoch: int,
        epoch_total: int,
        _lr: float,
    ) -> None:
        _submit_to_progress_loop(
            _update_progress(job_id, step, total, loss, avr_loss, epoch, epoch_total),
        )

    return callback


def _make_sampling_status_callback(job_id: int):
    def callback(status: str | None) -> None:
        _submit_to_progress_loop(_update_sampling_status(job_id, status))

    return callback


def _make_sampling_progress_callback(job_id: int):
    def callback(step: int, total: int) -> None:
        _submit_to_progress_loop(_update_sampling_progress(job_id, step, total))

    return callback


def _build_log_path(job_id: int) -> Path:
    logs_dir = Path(settings.training.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"job_{job_id}.log"


async def _run(job_id: int) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is None:
            logger.error("Job id=%d not found in DB", job_id)
            sys.exit(1)
        config_yaml = job.config_yaml

    config = TrainConfig.from_yaml(config_yaml)
    log_path = _build_log_path(job_id)
    metric_logger: MetricLogger | None = None
    if config.logging.use_ui_logger:
        loss_log_path = build_loss_log_path(config)
        reset_loss_log(loss_log_path)
        metric_logger = MetricLogger(loss_log_path)
    tensorboard_writer = None
    if config.logging.log_dir:
        tensorboard_writer, tensorboard_dir = setup_tensorboard_writer(
            config.logging.log_dir,
            job_id,
        )
    training_logger = JobTrainingLogger(
        job_id=job_id,
        log_path=log_path,
        metric_logger=metric_logger,
        log_every=config.logging.log_every,
        tensorboard_writer=tensorboard_writer,
    )
    if config.logging.log_dir:
        training_logger.logger.info("TensorBoard log dir: %s", tensorboard_dir)
    await _set_log_path(job_id, str(log_path))
    training_logger.logger.info(
        "Starting SDXL LoRA training for job id=%d: %s/%s", job_id, config.output_dir, config.lora_name
    )

    await _update_status(job_id, JobStatus.RUNNING)

    try:
        trainer = SDXLLoRATrainer(
            config,
            progress_callback=_make_progress_callback(job_id),
            sampling_status_callback=_make_sampling_status_callback(job_id),
            sampling_progress_callback=_make_sampling_progress_callback(job_id),
            training_logger=training_logger,
        )
        trainer.train()
        await _update_status(job_id, JobStatus.COMPLETED)
        training_logger.logger.info("Job id=%d completed successfully", job_id)
    except Exception as exc:
        training_logger.logger.exception("Job id=%d failed: %s", job_id, exc)
        await _update_status(job_id, JobStatus.FAILED, error=str(exc))
        sys.exit(1)
    finally:
        training_logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LoRA training job")
    parser.add_argument("--job-id", type=int, required=True, help="TrainingJob ID in the database")
    args = parser.parse_args()
    asyncio.run(_run(args.job_id))


if __name__ == "__main__":
    main()
