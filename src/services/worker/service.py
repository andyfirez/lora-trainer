"""Queue worker — polls SQLite and spawns training subprocesses."""

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass

import psutil

from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.session import session_factory
from src.db.tables.training_job import JobStatus
from src.settings.app_settings import settings

logger = logging.getLogger(__name__)


@dataclass
class _ManagedProcess:
    proc: subprocess.Popen[bytes]

    @property
    def pid(self) -> int | None:
        return self.proc.pid

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode

    def is_running(self) -> bool:
        return self.proc.poll() is None

    async def wait(self) -> int:
        return await asyncio.to_thread(self.proc.wait)


def _drain_subprocess_output(proc: subprocess.Popen[bytes]) -> None:
    if proc.stdout is None:
        return
    for _line in proc.stdout:
        pass


def _log_subprocess_output(proc: subprocess.Popen[bytes], job_id: int) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logger.info("[job %d] %s", job_id, line.decode(errors="replace").rstrip())


class QueueWorker:
    def __init__(self, *, echo_subprocess_output: bool = False) -> None:
        self._echo_subprocess_output = echo_subprocess_output
        self._active_jobs: dict[int, _ManagedProcess] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._cancel_task: asyncio.Task[None] | None = None
        self._job_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        logger.info(
            "Queue worker started — polling every %ds, max %d concurrent job(s)",
            settings.training.worker_poll_interval_seconds,
            settings.training.max_concurrent_jobs,
        )
        self._cancel_task = asyncio.create_task(self._watch_cancellations())
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._cancel_task is not None:
            self._cancel_task.cancel()
            try:
                await self._cancel_task
            except asyncio.CancelledError:
                pass
            self._cancel_task = None

        for job_id, managed in list(self._active_jobs.items()):
            if managed.is_running() and managed.pid is not None:
                logger.info("Shutting down — terminating job id=%d pid=%d", job_id, managed.pid)
                self._kill_process_tree(managed.pid)

        if self._job_tasks:
            await asyncio.gather(*self._job_tasks, return_exceptions=True)
        self._job_tasks.clear()
        logger.info("Queue worker stopped")

    def _kill_process_tree(self, pid: int) -> None:
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

    async def _is_any_job_running(self) -> bool:
        async with session_factory() as session:
            repo = TrainingJobRepository(session)
            running = await repo.get_running()
            return running is not None

    async def _get_next_queued_job_id(self) -> int | None:
        async with session_factory() as session:
            queue_repo = QueueRepository(session)
            next_entry = await queue_repo.get_next()
            return next_entry.job_id if next_entry is not None else None

    async def _mark_job_running(self, job_id: int, pid: int) -> None:
        async with session_factory() as session:
            repo = TrainingJobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is not None:
                await repo.update_status(job, JobStatus.RUNNING, pid=pid)
                await session.commit()

    async def _mark_job_spawn_failed(self, job_id: int, error_message: str) -> None:
        async with session_factory() as session:
            repo = TrainingJobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is not None and job.status != JobStatus.RUNNING:
                await repo.update_status(job, JobStatus.FAILED, error_message=error_message)
                await session.commit()

    async def _dequeue_job(self, job_id: int) -> None:
        async with session_factory() as session:
            queue_repo = QueueRepository(session)
            entry = await queue_repo.get_by_job_id(job_id)
            if entry is not None:
                await queue_repo.shift_positions_down(entry.position)
                await queue_repo.delete(entry)
                await session.commit()

    async def _get_job_status(self, job_id: int) -> JobStatus | None:
        async with session_factory() as session:
            repo = TrainingJobRepository(session)
            job = await repo.get_by_id(job_id)
            return job.status if job is not None else None

    async def _finalize_job(self, job_id: int, return_code: int) -> None:
        async with session_factory() as session:
            repo = TrainingJobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                return
            if job.status == JobStatus.CANCELLED:
                logger.info("Job id=%d finished after cancellation (exit code %d)", job_id, return_code)
                await repo.clear_process_state(job)
                await session.commit()
                return
            if job.status == JobStatus.RUNNING:
                status = JobStatus.COMPLETED if return_code == 0 else JobStatus.FAILED
                error_message = None if return_code == 0 else f"Training process exited with code {return_code}"
                await repo.update_status(job, status, error_message=error_message)
                await session.commit()
                logger.info("Job id=%d finished with status=%s (exit code %d)", job_id, status, return_code)

    async def _watch_cancellations(self) -> None:
        interval = settings.training.cancel_poll_interval_seconds
        while True:
            try:
                for job_id, managed in list(self._active_jobs.items()):
                    if not managed.is_running():
                        continue
                    status = await self._get_job_status(job_id)
                    if status == JobStatus.CANCELLED and managed.pid is not None:
                        logger.info("Cancellation requested for job id=%d, killing pid=%d", job_id, managed.pid)
                        self._kill_process_tree(managed.pid)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cancellation watcher error")
            await asyncio.sleep(interval)

    async def _run_job(self, job_id: int) -> None:
        await self._dequeue_job(job_id)
        managed: _ManagedProcess | None = None
        try:
            logger.info("Spawning trainer subprocess for job id=%d", job_id)
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.trainer.runner", "--job-id", str(job_id)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            managed = _ManagedProcess(proc)
            self._active_jobs[job_id] = managed
            if managed.pid is not None:
                await self._mark_job_running(job_id, managed.pid)

            if self._echo_subprocess_output:
                await asyncio.gather(
                    asyncio.to_thread(_log_subprocess_output, proc, job_id),
                    managed.wait(),
                )
            else:
                await asyncio.gather(
                    asyncio.to_thread(_drain_subprocess_output, proc),
                    managed.wait(),
                )
        except Exception as exc:
            logger.exception("Failed to spawn trainer for job id=%d", job_id)
            await self._mark_job_spawn_failed(job_id, str(exc))
        finally:
            self._active_jobs.pop(job_id, None)
            if managed is not None:
                await self._finalize_job(job_id, managed.returncode or 0)

    async def _poll_loop(self) -> None:
        interval = settings.training.worker_poll_interval_seconds
        while True:
            try:
                if not await self._is_any_job_running():
                    job_id = await self._get_next_queued_job_id()
                    if job_id is not None and job_id not in self._active_jobs:
                        task = asyncio.create_task(self._run_job(job_id))
                        self._job_tasks.add(task)
                        task.add_done_callback(self._job_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker poll error")
            await asyncio.sleep(interval)
