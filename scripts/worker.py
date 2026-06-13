"""Queue worker — polls SQLite and spawns training subprocesses.

Run with:
    uv run python -m scripts.worker
"""

import asyncio
import logging
import sys

import psutil

from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.session import create_tables, session_factory
from src.db.tables.training_job import JobStatus
from src.settings.app_settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_active_jobs: dict[int, asyncio.subprocess.Process] = {}


def _kill_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        logger.info("Terminated process tree for pid=%d", pid)
    except psutil.NoSuchProcess:
        logger.info("Process pid=%d already terminated", pid)
    except Exception as exc:
        logger.error("Failed to terminate process pid=%d: %s", pid, exc)


async def _is_any_job_running() -> bool:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        running = await repo.get_running()
        return running is not None


async def _get_next_queued_job_id() -> int | None:
    async with session_factory() as session:
        queue_repo = QueueRepository(session)
        next_entry = await queue_repo.get_next()
        return next_entry.job_id if next_entry is not None else None


async def _mark_job_running(job_id: int, pid: int) -> None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is not None:
            await repo.update_status(job, JobStatus.RUNNING, pid=pid)
            await session.commit()


async def _dequeue_job(job_id: int) -> None:
    async with session_factory() as session:
        queue_repo = QueueRepository(session)
        entry = await queue_repo.get_by_job_id(job_id)
        if entry is not None:
            await queue_repo.shift_positions_down(entry.position)
            await queue_repo.delete(entry)
            await session.commit()


async def _get_job_status(job_id: int) -> JobStatus | None:
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await repo.get_by_id(job_id)
        return job.status if job is not None else None


async def _finalize_job(job_id: int, return_code: int) -> None:
    await _dequeue_job(job_id)
    async with session_factory() as session:
        repo = TrainingJobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is None:
            return
        if job.status == JobStatus.CANCELLED:
            logger.info("Job id=%d finished after cancellation (exit code %d)", job_id, return_code)
            await session.commit()
            return
        if job.status == JobStatus.RUNNING:
            status = JobStatus.COMPLETED if return_code == 0 else JobStatus.FAILED
            await repo.update_status(job, status)
            await session.commit()
            logger.info("Job id=%d finished with status=%s (exit code %d)", job_id, status, return_code)


async def _watch_cancellations() -> None:
    interval = settings.training.cancel_poll_interval_seconds
    while True:
        try:
            for job_id, proc in list(_active_jobs.items()):
                if proc.returncode is not None:
                    continue
                status = await _get_job_status(job_id)
                if status == JobStatus.CANCELLED:
                    logger.info("Cancellation requested for job id=%d, killing pid=%d", job_id, proc.pid)
                    _kill_process_tree(proc.pid)
        except Exception:
            logger.exception("Cancellation watcher error")
        await asyncio.sleep(interval)


async def _run_job(job_id: int) -> None:
    logger.info("Spawning trainer subprocess for job id=%d", job_id)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "src.trainer.runner", "--job-id", str(job_id),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    _active_jobs[job_id] = proc
    await _mark_job_running(job_id, proc.pid)

    async def _stream_output() -> None:
        if proc.stdout is not None:
            async for line in proc.stdout:
                logger.info("[job %d] %s", job_id, line.decode(errors="replace").rstrip())

    try:
        await asyncio.gather(_stream_output(), proc.wait())
    finally:
        _active_jobs.pop(job_id, None)
        await _finalize_job(job_id, proc.returncode or 0)


async def _poll_loop() -> None:
    interval = settings.training.worker_poll_interval_seconds

    while True:
        try:
            if not await _is_any_job_running():
                job_id = await _get_next_queued_job_id()
                if job_id is not None:
                    asyncio.create_task(_run_job(job_id))
        except Exception:
            logger.exception("Worker poll error")
        await asyncio.sleep(interval)


async def _main() -> None:
    await create_tables()
    logger.info(
        "Worker started — polling every %ds, max %d concurrent job(s)",
        settings.training.worker_poll_interval_seconds,
        settings.training.max_concurrent_jobs,
    )
    asyncio.create_task(_watch_cancellations())
    await _poll_loop()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
