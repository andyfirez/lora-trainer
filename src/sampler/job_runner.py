"""Shared sampling job execution for the runner subprocess."""

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import yaml

from src.db.repositories.job_repo import JobRepository
from src.db.session import session_factory
from src.db.tables.job import Job, JobStatus, JobType
from src.sampler.config import SamplingConfig
from src.sampler.sdxl.service import SDXLLoRASampler
from src.services.jobs.job_logging import build_job_log_path, build_job_logger

logger = logging.getLogger(__name__)

_progress_loop: asyncio.AbstractEventLoop | None = None


def _ensure_progress_loop() -> asyncio.AbstractEventLoop:
    global _progress_loop, _progress_thread
    if _progress_loop is None:
        import threading

        _progress_loop = asyncio.new_event_loop()
        threading.Thread(
            target=_progress_loop.run_forever,
            daemon=True,
            name="sampling-progress-db-loop",
        ).start()
    assert _progress_loop is not None
    return _progress_loop


async def _get_active_job(repo: JobRepository, job_id: int) -> Job | None:
    job = await repo.get_by_id(job_id)
    if job is None or job.status == JobStatus.CANCELLED:
        return None
    return job


async def _update_status(job_id: int, status: JobStatus, error: str | None = None) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            if job.status == JobStatus.CANCELLED and status != JobStatus.CANCELLED:
                return
            await repo.update_status(job, status, error_message=error)
            await session.commit()


async def _update_progress_status(job_id: int, status: str | None) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await _get_active_job(repo, job_id)
        if job is not None:
            await repo.update_progress_status(job, status)
            await session.commit()


async def _update_progress(job_id: int, step: int, total: int) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await _get_active_job(repo, job_id)
        if job is not None:
            await repo.update_progress(job, step, total)
            await session.commit()


async def _set_log_path(job_id: int, log_path: str) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            await repo.update_log_path(job, log_path)
            await session.commit()


async def _set_output_path(job_id: int, output_path: str) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            await repo.update_output_path(job, output_path)
            await session.commit()


def _submit_to_progress_loop(coro: Coroutine[Any, Any, None]) -> None:
    loop = _ensure_progress_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)

    def _log_exception(fut: asyncio.Future[object]) -> None:
        try:
            fut.result()
        except Exception:
            logger.exception("Sampling progress DB update failed")

    future.add_done_callback(_log_exception)


def _make_progress_status_callback(job_id: int):
    def callback(status: str | None) -> None:
        _submit_to_progress_loop(_update_progress_status(job_id, status))

    return callback


def _make_progress_callback(job_id: int):
    def callback(step: int, total: int) -> None:
        _submit_to_progress_loop(_update_progress(job_id, step, total))

    return callback


async def _fail_job(job_id: int, run_logger: logging.Logger, message: str) -> None:
    run_logger.error(message)
    await _update_status(job_id, JobStatus.FAILED, error=message)


async def run_sampling_job(job_id: int) -> int:
    log_path = build_job_log_path(job_id)
    run_logger = build_job_logger(job_id, log_path, name_prefix="sampling-job")
    await _set_log_path(job_id, str(log_path))

    try:
        async with session_factory() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None or job.job_type != JobType.SAMPLING:
                await _fail_job(job_id, run_logger, f"Sampling job id={job_id} not found in DB")
                return 1
            if job.status == JobStatus.CANCELLED:
                run_logger.info("Sampling job id=%d already cancelled", job_id)
                return 1
            config_yaml = job.config_yaml
            lora_paths_yaml = job.lora_paths_yaml
            output_path = job.output_path

        sampling_config = SamplingConfig.from_yaml(config_yaml)
        train_config = sampling_config.to_train_config()
        train_config.validate_gpu()
        lora_paths = [Path(path) for path in (yaml.safe_load(lora_paths_yaml or "[]") or [])]
        if output_path is None:
            output_path = str(
                Path(sampling_config.output_dir)
                / sampling_config.lora_name
                / "samples"
                / f"job_{job_id}"
            )
            await _set_output_path(job_id, output_path)

        run_logger.info("Starting sampling job id=%d with %d LoRA file(s)", job_id, len(lora_paths))
        sampler = SDXLLoRASampler(
            train_config,
            lora_paths=lora_paths,
            output_dir=Path(output_path),
            progress_status_callback=_make_progress_status_callback(job_id),
            progress_callback=_make_progress_callback(job_id),
            log=run_logger,
        )
        sampler.run()
        await _update_progress_status(job_id, None)
        await _update_status(job_id, JobStatus.COMPLETED)
        run_logger.info("Sampling job id=%d completed successfully", job_id)
        return 0
    except Exception as exc:
        run_logger.exception("Sampling job id=%d failed: %s", job_id, exc)
        await _update_status(job_id, JobStatus.FAILED, error=str(exc))
        return 1
    finally:
        for handler in list(run_logger.handlers):
            handler.close()
            run_logger.removeHandler(handler)
