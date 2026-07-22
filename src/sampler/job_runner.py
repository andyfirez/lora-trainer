"""Shared sampling job execution for the runner subprocess."""

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import yaml
from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_repo import JobRepository
from src.db.session import session_factory
from src.db.tables.job import Job, JobStatus, JobType
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.sampler.sdxl.service import SDXLLoRASampler
from src.services.datasets.service import reconcile_datasets_for_training
from src.services.jobs.job_logging import build_job_log_path, build_job_logger
from src.services.jobs.sampling_jobs import prepare_sampling_config_lora_paths
from src.trainer.concept_training_metadata import (
    ConceptTrainingMetadata,
    resolve_concept_training_metadata,
)
from src.trainer.config import TrainConfig
from src.trainer.sdxl.caption import collect_trigger_words

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
            source_job_id = job.source_job_id

        sampling_config = SamplingConfig.from_yaml(config_yaml)
        job_lora_paths = [str(p) for p in (yaml.safe_load(lora_paths_yaml or "[]") or [])]
        default_trigger: str | None = None
        source_train_config: TrainConfig | None = None
        if source_job_id is not None:
            async with session_factory() as session:
                repo = JobRepository(session)
                source_job = await repo.get_by_id(source_job_id)
                if source_job is not None and source_job.job_type == JobType.TRAINING:
                    source_train_config = TrainConfig.from_yaml(source_job.config_yaml)
        if source_train_config is not None:
            words = collect_trigger_words(source_train_config.concepts)
            if words:
                default_trigger = ", ".join(words)

        sampling_config, effective_lora_paths = prepare_sampling_config_lora_paths(
            sampling_config,
            job_lora_paths or None,
            default_trigger=default_trigger,
        )
        if effective_lora_paths and not job_lora_paths:
            run_logger.info(
                "LoRA paths taken from sampling config (%d file(s), job record had none)",
                len(effective_lora_paths),
            )
        lora_paths = [Path(p) for p in effective_lora_paths]
        train_config = sampling_config.to_train_config()
        train_config.validate_gpu()
        if source_train_config is not None:
            train_config = train_config.model_copy(
                update={"clip_skip": source_train_config.clip_skip},
            )
        if output_path is None:
            output_path = str(
                resolve_sampling_output_path(sampling_config, job_id, source_train_config)
            )
            await _set_output_path(job_id, output_path)

        concept_metadata: dict[int, ConceptTrainingMetadata] | None = None
        if source_train_config is not None:
            dataset_ids = [c.dataset_id for c in source_train_config.concepts]
            async with session_factory() as session:
                dataset_repo = DatasetRepository(session)
                crop_repo = DatasetImageCropRepository(session)
                await reconcile_datasets_for_training(dataset_ids, dataset_repo, crop_repo)
                concept_metadata = await resolve_concept_training_metadata(
                    dataset_ids,
                    dataset_repo,
                    crop_repo,
                )

        from src.sampler.sweep.combinations import count_combinations

        combo_count = count_combinations(sampling_config.parameters)
        run_logger.info(
            "Starting sampling job id=%d: %d LoRA file(s), %d sweep image(s)",
            job_id,
            len(lora_paths),
            combo_count,
        )
        if lora_paths:
            for index, path in enumerate(lora_paths, start=1):
                run_logger.info("  LoRA %d/%d: %s", index, len(lora_paths), path)
        vary_keys = sampling_config.parameters.vary_keys_with_values()
        if vary_keys:
            run_logger.info("  Varying parameters: %s", ", ".join(vary_keys))
        sampler = SDXLLoRASampler(
            train_config,
            sampling_config=sampling_config,
            lora_paths=lora_paths,
            output_dir=Path(output_path),
            progress_status_callback=_make_progress_status_callback(job_id),
            progress_callback=_make_progress_callback(job_id),
            log=run_logger,
            concept_metadata=concept_metadata,
            job_id=job_id,
            compose_grids=True,
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
