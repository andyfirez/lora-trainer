"""Tagging runner — CLI entry point spawned by the queue worker."""

import argparse
import asyncio
import logging
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_repo import JobRepository
from src.db.session import session_factory
from src.db.tables.job import Job, JobStatus
from src.services.datasets.captions import list_image_filenames, merge_tags, read_tags, write_tags
from src.services.datasets.training_cache import invalidate_te_cache_for_image
from src.services.jobs.job_logging import build_job_log_path, build_job_logger
from src.tagger.config import TaggingConfig
from src.tagger.wd14 import WD14Tagger

logger = logging.getLogger(__name__)

_progress_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
threading.Thread(
    target=_progress_loop.run_forever,
    daemon=True,
    name="tagging-progress-db-loop",
).start()


async def _get_active_job(repo: JobRepository, job_id: int) -> Job | None:
    job = await repo.get_by_id(job_id)
    if job is None or job.status == JobStatus.CANCELLED:
        return None
    return job


async def _update_progress(job_id: int, step: int, total: int, status: str) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await _get_active_job(repo, job_id)
        if job is not None:
            await repo.update_progress(job, step, total)
            await repo.update_progress_status(job, status)
            await session.commit()


async def _update_status(job_id: int, status: JobStatus, error: str | None = None) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            if job.status == JobStatus.CANCELLED and status != JobStatus.CANCELLED:
                return
            await repo.update_status(job, status, error_message=error)
            await session.commit()


async def _set_log_path(job_id: int, log_path: str) -> None:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            await repo.update_log_path(job, log_path)
            await session.commit()


def _submit_to_progress_loop(coro: Coroutine[object, object, None]) -> None:
    asyncio.run_coroutine_threadsafe(coro, _progress_loop)


async def _load_job(job_id: int) -> Job:
    async with session_factory() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is None:
            raise RuntimeError(f"Job id={job_id} not found")
        return job


def _resolve_targets(config: TaggingConfig) -> list[str]:
    image_dir = Path(config.image_dir)
    if config.filenames:
        return list(config.filenames)
    return list_image_filenames(image_dir)


async def _load_dataset_target_resolution(dataset_id: int) -> int | None:
    async with session_factory() as session:
        repo = DatasetRepository(session)
        dataset = await repo.get_by_id(dataset_id)
        if dataset is None:
            return None
        return dataset.target_resolution


async def run_tagging_job(job_id: int) -> int:
    log_path = build_job_log_path(job_id)
    run_logger = build_job_logger(job_id, log_path, name_prefix="tagging-job")
    await _set_log_path(job_id, str(log_path))

    job = await _load_job(job_id)
    config = TaggingConfig.from_yaml(job.config_yaml)
    target_resolution = await _load_dataset_target_resolution(config.dataset_id)
    targets = _resolve_targets(config)
    total = len(targets)

    if total == 0:
        run_logger.warning("No images to tag for job id=%d", job_id)
        await _update_status(job_id, JobStatus.COMPLETED)
        return 0

    image_dir = Path(config.image_dir)
    model_repo = config.resolve_model_repo()
    run_logger.info(
        "Starting tagging job id=%d: %d image(s), model=%s, mode=%s, threshold=%.2f",
        job_id,
        total,
        config.model,
        config.mode.value,
        config.threshold,
    )
    run_logger.info("Loading WD14 model from %s", model_repo)
    tagger = WD14Tagger(model_repo)
    run_logger.info("Model loaded")

    for index, filename in enumerate(targets, start=1):
        async with session_factory() as session:
            repo = JobRepository(session)
            active = await _get_active_job(repo, job_id)
            if active is None:
                run_logger.info("Tagging job id=%d cancelled", job_id)
                return 1

        image_path = image_dir / filename
        if not image_path.is_file():
            run_logger.warning("Skipping missing image: %s", filename)
            _submit_to_progress_loop(_update_progress(job_id, index, total, f"Skipped {filename}"))
            continue

        predicted = tagger.predict(
            image_path,
            threshold=config.threshold,
            strip_rating=config.strip_rating,
        )
        existing = read_tags(image_dir, filename, config.caption_extension)
        merged = merge_tags(existing, predicted, config.mode.value)
        write_tags(image_dir, filename, merged, config.caption_extension)
        invalidate_te_cache_for_image(image_dir, filename, target_resolution)
        run_logger.info(
            "Tagged %s (%d/%d): %d tag(s) -> %d total",
            filename,
            index,
            total,
            len(predicted),
            len(merged),
        )
        _submit_to_progress_loop(
            _update_progress(job_id, index, total, f"Tagged {filename} ({index}/{total})")
        )

    await _update_progress(job_id, total, total, "Completed")
    await _update_status(job_id, JobStatus.COMPLETED)
    run_logger.info("Tagging job id=%d completed", job_id)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dataset tagging job")
    parser.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(run_tagging_job(args.job_id))
        if exit_code != 0:
            sys.exit(exit_code)
    except SystemExit:
        raise
    except BaseException as exc:
        logger.exception("Tagging runner failed: %s", exc)
        asyncio.run(_update_status(args.job_id, JobStatus.FAILED, str(exc)))
        sys.exit(1)


if __name__ == "__main__":
    main()
