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

from src.db.repositories.sampling_run_repo import SamplingRunRepository
from src.db.session import session_factory
from src.db.tables.sampling_run import SamplingRun, SamplingRunStatus
from src.settings.app_settings import settings
from src.trainer.config import TrainConfig

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


async def _get_active_run(repo: SamplingRunRepository, sampling_run_id: int) -> SamplingRun | None:
    sampling_run = await repo.get_by_id(sampling_run_id)
    if sampling_run is None or sampling_run.status == SamplingRunStatus.CANCELLED:
        return None
    return sampling_run


async def _update_status(
    sampling_run_id: int,
    status: SamplingRunStatus,
    error: str | None = None,
) -> None:
    async with session_factory() as session:
        repo = SamplingRunRepository(session)
        sampling_run = await repo.get_by_id(sampling_run_id)
        if sampling_run is not None:
            if sampling_run.status == SamplingRunStatus.CANCELLED and status != SamplingRunStatus.CANCELLED:
                return
            await repo.update_status(sampling_run, status, error_message=error)
            await session.commit()


async def _update_progress_status(sampling_run_id: int, status: str | None) -> None:
    async with session_factory() as session:
        repo = SamplingRunRepository(session)
        sampling_run = await _get_active_run(repo, sampling_run_id)
        if sampling_run is not None:
            await repo.update_progress_status(sampling_run, status)
            await session.commit()


async def _update_progress(sampling_run_id: int, step: int, total: int) -> None:
    async with session_factory() as session:
        repo = SamplingRunRepository(session)
        sampling_run = await _get_active_run(repo, sampling_run_id)
        if sampling_run is not None:
            await repo.update_progress(sampling_run, step, total)
            await session.commit()


async def _set_log_path(sampling_run_id: int, log_path: str) -> None:
    async with session_factory() as session:
        repo = SamplingRunRepository(session)
        sampling_run = await repo.get_by_id(sampling_run_id)
        if sampling_run is not None:
            await repo.update_log_path(sampling_run, log_path)
            await session.commit()


async def _set_output_path(sampling_run_id: int, output_path: str) -> None:
    async with session_factory() as session:
        repo = SamplingRunRepository(session)
        sampling_run = await repo.get_by_id(sampling_run_id)
        if sampling_run is not None:
            await repo.update_output_path(sampling_run, output_path)
            await session.commit()


def _submit_to_progress_loop(coro: Coroutine[Any, Any, None]) -> None:
    future = asyncio.run_coroutine_threadsafe(coro, _progress_loop)

    def _log_exception(fut: asyncio.Future[object]) -> None:
        try:
            fut.result()
        except Exception:
            logger.exception("Sampling progress DB update failed")

    future.add_done_callback(_log_exception)


def _make_progress_status_callback(sampling_run_id: int):
    def callback(status: str | None) -> None:
        _submit_to_progress_loop(_update_progress_status(sampling_run_id, status))

    return callback


def _make_progress_callback(sampling_run_id: int):
    def callback(step: int, total: int) -> None:
        _submit_to_progress_loop(_update_progress(sampling_run_id, step, total))

    return callback


def _build_log_path(sampling_run_id: int) -> Path:
    logs_dir = Path(settings.training.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"sampling_run_{sampling_run_id}.log"


def _build_logger(sampling_run_id: int, log_path: Path) -> logging.Logger:
    run_logger = logging.getLogger(f"sampling-run-{sampling_run_id}")
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


async def _fail_and_exit(
    sampling_run_id: int,
    run_logger: logging.Logger,
    message: str,
) -> None:
    run_logger.error(message)
    await _update_status(sampling_run_id, SamplingRunStatus.FAILED, error=message)
    sys.exit(1)


async def _run(sampling_run_id: int) -> None:
    log_path = _build_log_path(sampling_run_id)
    run_logger = _build_logger(sampling_run_id, log_path)
    await _set_log_path(sampling_run_id, str(log_path))

    try:
        async with session_factory() as session:
            repo = SamplingRunRepository(session)
            sampling_run = await repo.get_by_id(sampling_run_id)
            if sampling_run is None:
                await _fail_and_exit(
                    sampling_run_id,
                    run_logger,
                    f"Sampling run id={sampling_run_id} not found in DB",
                )
            config_yaml = sampling_run.config_yaml
            lora_paths_yaml = sampling_run.lora_paths_yaml
            output_path = sampling_run.output_path

        config = TrainConfig.from_yaml(config_yaml)
        lora_paths = [Path(path) for path in (yaml.safe_load(lora_paths_yaml) or [])]
        if output_path is None:
            output_path = str(
                Path(config.output_dir) / config.lora_name / "samples" / f"sampling_run_{sampling_run_id}"
            )
            await _set_output_path(sampling_run_id, output_path)

        run_logger.info("Starting sampling run id=%d with %d LoRA file(s)", sampling_run_id, len(lora_paths))
        from src.sampler.sdxl.service import SDXLLoRASampler

        sampler = SDXLLoRASampler(
            config,
            lora_paths=lora_paths,
            output_dir=Path(output_path),
            progress_status_callback=_make_progress_status_callback(sampling_run_id),
            progress_callback=_make_progress_callback(sampling_run_id),
            log=run_logger,
        )
        sampler.run()
        await _update_progress_status(sampling_run_id, None)
        await _update_status(sampling_run_id, SamplingRunStatus.COMPLETED)
        run_logger.info("Sampling run id=%d completed successfully", sampling_run_id)
    except SystemExit:
        raise
    except Exception as exc:
        run_logger.exception("Sampling run id=%d failed: %s", sampling_run_id, exc)
        await _update_status(sampling_run_id, SamplingRunStatus.FAILED, error=str(exc))
        sys.exit(1)
    finally:
        for handler in list(run_logger.handlers):
            handler.close()
            run_logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standalone LoRA sampling job")
    parser.add_argument("--sampling-run-id", type=int, required=True, help="SamplingRun ID in the database")
    args = parser.parse_args()
    log_path = _build_log_path(args.sampling_run_id)
    run_logger = _build_logger(args.sampling_run_id, log_path)
    try:
        asyncio.run(_run(args.sampling_run_id))
    except SystemExit:
        raise
    except BaseException as exc:
        run_logger.exception("Sampling runner failed before run completed: %s", exc)
        try:
            asyncio.run(
                _update_status(args.sampling_run_id, SamplingRunStatus.FAILED, error=str(exc)),
            )
            asyncio.run(_set_log_path(args.sampling_run_id, str(log_path)))
        except Exception:
            logger.exception("Failed to persist sampling run failure status")
        sys.exit(1)
    finally:
        for handler in list(run_logger.handlers):
            handler.close()
            run_logger.removeHandler(handler)


if __name__ == "__main__":
    main()
