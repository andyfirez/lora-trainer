"""Training runner — CLI entry point spawned by the runnable worker.

Usage:
    python -m src.trainer.runner --lora-id <id>

Loads the Lora's config_yaml from SQLite, runs the SDXL LoRA trainer, and
writes progress back to the DB synchronously. Final status (completed/failed)
is derived by the worker from this process's exit code — this runner only
sets status explicitly for a graceful cancellation (stop-after-save/cache).
"""

import argparse
import asyncio
import logging
import sys
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.session import session_factory
from src.db.tables.lora import Lora
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.datasets.service import reconcile_datasets_for_training
from src.services.datasets.training_validation import validate_dataset_for_training
from src.services.runnable import runtime
from src.services.worker.progress_loop import (
    run_in_progress_loop,
    start_progress_loop,
    submit_to_progress_loop,
)
from src.settings.app_settings import settings
from src.storage.paths import StoragePaths
from src.trainer.concept_resolution import resolve_concept_paths
from src.trainer.concept_training_metadata import resolve_concept_training_metadata
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import MetricLogger, build_loss_log_path, reset_loss_log
from src.trainer.sdxl.trainer import (
    SDXLLoRATrainer,
    TrainingCancelledAfterSave,
    TrainingCancelledDuringCache,
)
from src.trainer.training_log import JobTrainingLogger, setup_tensorboard_writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_progress_loop = start_progress_loop("progress-db-loop")


async def _get_active_lora(repo: LoraRepository, lora_id: int) -> Lora | None:
    lora = await repo.get_by_id(lora_id)
    if lora is None or lora.status == RunnableStatus.CANCELLED:
        return None
    return lora


async def _update_progress(
    lora_id: int,
    step: int,
    total: int,
    loss: float,
    avr_loss: float,
    epoch: int,
    epoch_total: int,
) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await _get_active_lora(repo, lora_id)
        if lora is not None:
            lora.progress_step = step
            lora.progress_total = total
            lora.progress_loss = loss
            lora.progress_avr_loss = avr_loss
            lora.progress_epoch = epoch
            lora.progress_epoch_total = epoch_total
            session.add(lora)
            await session.commit()


async def _mark_cancelled(lora_id: int) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is not None and lora.status != RunnableStatus.CANCELLED:
            runtime.cancel(lora)
            lora.save_checkpoint_requested = False
            session.add(lora)
            await session.commit()


async def _set_log_path(lora_id: int, log_path: str) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is not None:
            lora.log_path = log_path
            session.add(lora)
            await session.commit()


async def _set_output_path(lora_id: int, output_path: str) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is not None:
            lora.output_path = output_path
            session.add(lora)
            await session.commit()


async def _update_checkpoint_info(lora_id: int, checkpoint_path: str, epoch: int, step: int) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is not None:
            lora.last_checkpoint_path = checkpoint_path
            lora.last_checkpoint_epoch = epoch
            lora.last_checkpoint_step = step
            session.add(lora)
            await session.commit()


async def _clear_resume_state(lora_id: int) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is not None:
            lora.resume_checkpoint_path = None
            lora.resume_from_epoch = None
            lora.resume_from_step = None
            session.add(lora)
            await session.commit()


async def _consume_save_checkpoint_request(lora_id: int) -> bool:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is None or not lora.save_checkpoint_requested:
            return False
        lora.save_checkpoint_requested = False
        session.add(lora)
        await session.commit()
        return True


async def _is_stop_requested(lora_id: int) -> bool:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is None or lora.status == RunnableStatus.CANCELLED:
            return True
        return lora.save_checkpoint_requested


def _submit_to_progress_loop(coro: Any) -> None:
    submit_to_progress_loop(_progress_loop, coro)


def _run_in_progress_loop(coro: Any, timeout_s: float = 1.0) -> Any:
    return run_in_progress_loop(_progress_loop, coro, timeout_s=timeout_s)


def _make_progress_callback(lora_id: int):
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
            _update_progress(lora_id, step, total, loss, avr_loss, epoch, epoch_total),
        )

    return callback


def _build_log_path(lora_id: int) -> Path:
    logs_dir = Path(settings.training.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"lora_{lora_id}.log"


async def _run(lora_id: int) -> None:
    async with session_factory() as session:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(lora_id)
        if lora is None:
            logger.error("Lora id=%d not found in DB", lora_id)
            sys.exit(1)
        config_yaml = lora.config_yaml
        resume_checkpoint_path = lora.resume_checkpoint_path

    if not config_yaml:
        logger.error("Lora id=%d has no config_yaml", lora_id)
        sys.exit(1)

    config = TrainConfig.from_snapshot_yaml(config_yaml)
    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        crop_repo = DatasetImageCropRepository(session)
        dataset_ids = [concept.dataset_id for concept in config.concepts]
        await reconcile_datasets_for_training(dataset_ids, dataset_repo, crop_repo)
        concept_paths = await resolve_concept_paths(dataset_ids, dataset_repo)
        concept_metadata = await resolve_concept_training_metadata(
            dataset_ids,
            dataset_repo,
            crop_repo,
        )
        for concept in config.concepts:
            dataset = await dataset_repo.get_by_id(concept.dataset_id)
            if dataset is None:
                logger.error("Dataset id=%d not found", concept.dataset_id)
                sys.exit(1)
            crops = await crop_repo.list_by_dataset(concept.dataset_id)
            try:
                validate_dataset_for_training(
                    dataset,
                    config.resolution,
                    enable_bucket=config.enable_bucket,
                    crops=list(crops),
                )
            except Exception as exc:
                logger.error("Dataset validation failed: %s", exc)
                sys.exit(1)
        config = config.resolve_concepts(concept_paths)
    if resume_checkpoint_path:
        config.resume_from_checkpoint = resume_checkpoint_path
    is_resume_run = bool(config.resume_from_checkpoint)
    log_path = _build_log_path(lora_id)
    metric_logger: MetricLogger | None = None
    if config.logging.use_ui_logger:
        loss_log_path = build_loss_log_path(config)
        if not is_resume_run:
            reset_loss_log(loss_log_path)
        metric_logger = MetricLogger(loss_log_path)
    tensorboard_writer = None
    if config.logging.log_dir:
        tensorboard_writer, tensorboard_dir = setup_tensorboard_writer(
            config.logging.log_dir,
            lora_id,
            reset_dir=not is_resume_run,
        )
    training_logger = JobTrainingLogger(
        job_id=lora_id,
        log_path=log_path,
        metric_logger=metric_logger,
        log_every=config.logging.log_every,
        tensorboard_writer=tensorboard_writer,
        append_log=is_resume_run,
    )
    if config.logging.log_dir:
        training_logger.logger.info("TensorBoard log dir: %s", tensorboard_dir)
    await _set_log_path(lora_id, str(log_path))
    work_dir = StoragePaths.resolve_training_work_dir(config.output_dir, config.lora_name)
    await _set_output_path(lora_id, str(work_dir))
    if not is_resume_run:
        await _clear_resume_state(lora_id)
    training_logger.logger.info(
        "Starting SDXL LoRA training for lora id=%d: %s/%s", lora_id, config.output_dir, config.lora_name
    )

    try:
        def _checkpoint_callback(path: str, epoch: int, step: int) -> None:
            _submit_to_progress_loop(_update_checkpoint_info(lora_id, path, epoch, step))

        def _save_checkpoint_requested() -> bool:
            try:
                return bool(_run_in_progress_loop(_consume_save_checkpoint_request(lora_id)))
            except FutureTimeoutError:
                return False
            except Exception:
                logger.exception("Failed to poll save-checkpoint request")
                return False

        def _stop_requested() -> bool:
            try:
                return bool(_run_in_progress_loop(_is_stop_requested(lora_id)))
            except FutureTimeoutError:
                return False
            except Exception:
                logger.exception("Failed to poll stop request during cache")
                return False

        trainer = SDXLLoRATrainer(
            config,
            progress_callback=_make_progress_callback(lora_id),
            training_logger=training_logger,
            checkpoint_callback=_checkpoint_callback,
            save_checkpoint_requested_callback=_save_checkpoint_requested,
            stop_requested_callback=_stop_requested,
            concept_metadata=concept_metadata,
        )
        trainer.train()
        training_logger.logger.info("Lora id=%d completed successfully", lora_id)
    except TrainingCancelledAfterSave:
        await _mark_cancelled(lora_id)
        training_logger.logger.info("Lora id=%d cancelled after saving checkpoint", lora_id)
    except TrainingCancelledDuringCache:
        await _mark_cancelled(lora_id)
        training_logger.logger.info("Lora id=%d cancelled during cache/setup", lora_id)
    except Exception as exc:
        training_logger.logger.exception("Lora id=%d failed: %s", lora_id, exc)
        sys.exit(1)
    finally:
        training_logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LoRA training run")
    parser.add_argument("--lora-id", type=int, required=True, help="Lora ID in the database")
    args = parser.parse_args()
    try:
        asyncio.run(_run(args.lora_id))
    except Exception:
        logger.exception("Unhandled error in training runner for lora id=%d", args.lora_id)
        sys.exit(1)


if __name__ == "__main__":
    main()
