"""Shared sampling execution for the runner subprocess."""

import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import yaml
from src.db.repositories.sampling_repo import SamplingRepository
from src.db.session import session_factory
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.sampler.sdxl.service import SDXLLoRASampler
from src.services.runnable.logging import build_runnable_log_path, build_runnable_logger
from src.services.sampling.lora_paths import prepare_sampling_config_lora_paths
from src.services.worker.progress_loop import (
    start_progress_loop,
    submit_to_progress_loop,
)

logger = logging.getLogger(__name__)

_progress_loop = start_progress_loop("sampling-progress-db-loop")


async def _get_active_sampling(repo: SamplingRepository, sampling_id: int) -> Sampling | None:
    sampling = await repo.get_by_id(sampling_id)
    if sampling is None or sampling.status == RunnableStatus.CANCELLED:
        return None
    return sampling


async def _update_progress_status(sampling_id: int, status: str | None) -> None:
    async with session_factory() as session:
        repo = SamplingRepository(session)
        sampling = await _get_active_sampling(repo, sampling_id)
        if sampling is not None:
            sampling.progress_status = status
            session.add(sampling)
            await session.commit()


async def _update_progress(sampling_id: int, step: int, total: int) -> None:
    async with session_factory() as session:
        repo = SamplingRepository(session)
        sampling = await _get_active_sampling(repo, sampling_id)
        if sampling is not None:
            sampling.progress_step = step
            sampling.progress_total = total
            session.add(sampling)
            await session.commit()


async def _set_output_path(sampling_id: int, output_path: str) -> None:
    async with session_factory() as session:
        repo = SamplingRepository(session)
        sampling = await repo.get_by_id(sampling_id)
        if sampling is not None:
            sampling.output_path = output_path
            session.add(sampling)
            await session.commit()


def _submit_to_progress_loop(coro: Coroutine[Any, Any, None]) -> None:
    submit_to_progress_loop(_progress_loop, coro)


def _make_progress_status_callback(sampling_id: int):
    def callback(status: str | None) -> None:
        _submit_to_progress_loop(_update_progress_status(sampling_id, status))

    return callback


def _make_progress_callback(sampling_id: int):
    def callback(step: int, total: int) -> None:
        _submit_to_progress_loop(_update_progress(sampling_id, step, total))

    return callback


async def run_sampling(sampling_id: int) -> int:
    """Run a sampling by id. Returns a process exit code; the worker derives
    final status (completed/failed) from it via the RunnableHandler."""
    log_path = build_runnable_log_path(sampling_id, prefix="sampling")
    run_logger = build_runnable_logger(sampling_id, log_path, name_prefix="sampling")

    try:
        async with session_factory() as session:
            repo = SamplingRepository(session)
            sampling = await repo.get_by_id(sampling_id)
            if sampling is None:
                run_logger.error("Sampling id=%d not found in DB", sampling_id)
                return 1
            if sampling.status == RunnableStatus.CANCELLED:
                run_logger.info("Sampling id=%d already cancelled", sampling_id)
                return 1
            sampling.log_path = str(log_path)
            session.add(sampling)
            await session.commit()
            config_yaml = sampling.config_yaml
            lora_paths_yaml = sampling.lora_paths_yaml
            output_path = sampling.output_path

        sampling_config = SamplingConfig.from_snapshot_yaml(config_yaml)
        stored_lora_paths = [str(p) for p in (yaml.safe_load(lora_paths_yaml or "[]") or [])]
        sampling_config, effective_lora_paths = prepare_sampling_config_lora_paths(
            sampling_config,
            stored_lora_paths or None,
        )
        if effective_lora_paths and not stored_lora_paths:
            run_logger.info(
                "LoRA paths taken from sampling config (%d file(s), record had none)",
                len(effective_lora_paths),
            )
        lora_paths = [Path(p) for p in effective_lora_paths]
        train_config = sampling_config.to_train_config()
        train_config.validate_gpu()
        if output_path is None:
            output_path = str(resolve_sampling_output_path(sampling_config, sampling_id))
            await _set_output_path(sampling_id, output_path)

        from src.sampler.sweep.combinations import count_combinations

        combo_count = count_combinations(sampling_config.parameters)
        run_logger.info(
            "Starting sampling id=%d: %d LoRA file(s), %d sweep image(s)",
            sampling_id,
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
            progress_status_callback=_make_progress_status_callback(sampling_id),
            progress_callback=_make_progress_callback(sampling_id),
            log=run_logger,
            concept_metadata=None,
            sampling_id=sampling_id,
            compose_grids=True,
        )
        sampler.run()
        await _update_progress_status(sampling_id, None)
        run_logger.info("Sampling id=%d completed successfully", sampling_id)
        return 0
    except Exception as exc:
        run_logger.exception("Sampling id=%d failed: %s", sampling_id, exc)
        return 1
    finally:
        for handler in list(run_logger.handlers):
            handler.close()
            run_logger.removeHandler(handler)
