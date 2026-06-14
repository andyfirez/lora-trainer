"""Sampling runner — CLI entry point spawned by the queue worker."""

import argparse
import asyncio
import logging
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import yaml

from src.db.repositories.job_repo import JobRepository
from src.db.session import session_factory
from src.db.tables.job import Job, JobStatus, JobType
from src.sampler.config import SamplingConfig
from src.settings.app_settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_progress_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
threading.Thread(
    target=_progress_loop.run_forever,
    daemon=True,
    name="sampling-progress-db-loop",
).start()


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
    future = asyncio.run_coroutine_threadsafe(coro, _progress_loop)

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


def _build_log_path(job_id: int) -> Path:
    logs_dir = Path(settings.training.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"job_{job_id}.log"


def _build_logger(job_id: int, log_path: Path) -> logging.Logger:
    run_logger = logging.getLogger(f"sampling-job-{job_id}")
    run_logger.setLevel(logging.INFO)
    run_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    run_logger.addHandler(file_handler)
    run_logger.addHandler(stream_handler)
    run_logger.propagate = False
    return run_logger


async def _fail_and_exit(job_id: int, run_logger: logging.Logger, message: str) -> None:
    run_logger.error(message)
    await _update_status(job_id, JobStatus.FAILED, error=message)
    sys.exit(1)


async def _run(job_id: int) -> None:
    log_path = _build_log_path(job_id)
    run_logger = _build_logger(job_id, log_path)
    await _set_log_path(job_id, str(log_path))

    try:
        async with session_factory() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None or job.job_type != JobType.SAMPLING:
                await _fail_and_exit(job_id, run_logger, f"Sampling job id={job_id} not found in DB")
            config_yaml = job.config_yaml
            lora_paths_yaml = job.lora_paths_yaml
            output_path = job.output_path

        sampling_config = SamplingConfig.from_yaml(config_yaml)
        train_config = sampling_config.to_train_config()
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
        from src.sampler.sdxl.service import SDXLLoRASampler

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
    except SystemExit:
        raise
    except Exception as exc:
        run_logger.exception("Sampling job id=%d failed: %s", job_id, exc)
        await _update_status(job_id, JobStatus.FAILED, error=str(exc))
        sys.exit(1)
    finally:
        for handler in list(run_logger.handlers):
            handler.close()
            run_logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standalone LoRA sampling job")
    parser.add_argument("--job-id", type=int, required=True, help="Job ID in the database")
    args = parser.parse_args()
    log_path = _build_log_path(args.job_id)
    run_logger = _build_logger(args.job_id, log_path)
    try:
        asyncio.run(_run(args.job_id))
    except SystemExit:
        raise
    except BaseException as exc:
        run_logger.exception("Sampling runner failed before run completed: %s", exc)
        try:
            asyncio.run(_update_status(args.job_id, JobStatus.FAILED, error=str(exc)))
            asyncio.run(_set_log_path(args.job_id, str(log_path)))
        except Exception:
            logger.exception("Failed to persist sampling job failure status")
        sys.exit(1)
    finally:
        for handler in list(run_logger.handlers):
            handler.close()
            run_logger.removeHandler(handler)


if __name__ == "__main__":
    main()
