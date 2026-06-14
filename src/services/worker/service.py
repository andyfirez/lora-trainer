"""Queue worker — polls SQLite and spawns training subprocesses."""

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass

import psutil

from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.sampling_run_repo import SamplingRunRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.session import session_factory
from src.db.tables.queue_entry import QueueEntry, QueueItemType
from src.db.tables.sampling_run import SamplingRunStatus
from src.db.tables.training_job import JobStatus
from src.settings.app_settings import settings
from src.services.sampling.exceptions import SamplingCheckpointsNotFoundError
from src.services.sampling.service import SamplingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QueueItemKey:
    item_type: QueueItemType
    item_id: int


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


def _log_subprocess_output(proc: subprocess.Popen[bytes], label: str) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logger.info("[%s] %s", label, line.decode(errors="replace").rstrip())


class QueueWorker:
    def __init__(self, *, echo_subprocess_output: bool = False) -> None:
        self._echo_subprocess_output = echo_subprocess_output
        self._active_jobs: dict[_QueueItemKey, _ManagedProcess] = {}
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

        for key, managed in list(self._active_jobs.items()):
            if managed.is_running() and managed.pid is not None:
                logger.info(
                    "Shutting down — terminating %s id=%d pid=%d",
                    key.item_type,
                    key.item_id,
                    managed.pid,
                )
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
            job_repo = TrainingJobRepository(session)
            sampling_run_repo = SamplingRunRepository(session)
            running_job = await job_repo.get_running()
            running_sampling_run = await sampling_run_repo.get_running()
            return running_job is not None or running_sampling_run is not None

    async def _get_next_queued_entry(self) -> QueueEntry | None:
        async with session_factory() as session:
            queue_repo = QueueRepository(session)
            return await queue_repo.get_next()

    async def _mark_item_running(self, key: _QueueItemKey, pid: int) -> None:
        async with session_factory() as session:
            if key.item_type == QueueItemType.TRAINING:
                repo = TrainingJobRepository(session)
                job = await repo.get_by_id(key.item_id)
                if job is not None:
                    await repo.update_status(job, JobStatus.RUNNING, pid=pid)
            elif key.item_type == QueueItemType.SAMPLING:
                repo = SamplingRunRepository(session)
                sampling_run = await repo.get_by_id(key.item_id)
                if sampling_run is not None:
                    await repo.update_status(sampling_run, SamplingRunStatus.RUNNING, pid=pid)
            await session.commit()

    async def _mark_item_spawn_failed(self, key: _QueueItemKey, error_message: str) -> None:
        async with session_factory() as session:
            if key.item_type == QueueItemType.TRAINING:
                repo = TrainingJobRepository(session)
                job = await repo.get_by_id(key.item_id)
                if job is not None and job.status != JobStatus.RUNNING:
                    await repo.update_status(job, JobStatus.FAILED, error_message=error_message)
            elif key.item_type == QueueItemType.SAMPLING:
                repo = SamplingRunRepository(session)
                sampling_run = await repo.get_by_id(key.item_id)
                if sampling_run is not None and sampling_run.status != SamplingRunStatus.RUNNING:
                    await repo.update_status(sampling_run, SamplingRunStatus.FAILED, error_message=error_message)
            await session.commit()

    async def _dequeue_entry(self, entry_id: int) -> None:
        async with session_factory() as session:
            queue_repo = QueueRepository(session)
            entry = await queue_repo.get_by_id(entry_id)
            if entry is not None:
                await queue_repo.shift_positions_down(entry.position)
                await queue_repo.delete(entry)
                await session.commit()

    async def _get_item_cancelled(self, key: _QueueItemKey) -> bool:
        async with session_factory() as session:
            if key.item_type == QueueItemType.TRAINING:
                repo = TrainingJobRepository(session)
                job = await repo.get_by_id(key.item_id)
                return job is not None and job.status == JobStatus.CANCELLED
            if key.item_type == QueueItemType.SAMPLING:
                repo = SamplingRunRepository(session)
                sampling_run = await repo.get_by_id(key.item_id)
                return sampling_run is not None and sampling_run.status == SamplingRunStatus.CANCELLED
            return False

    async def _finalize_item(self, key: _QueueItemKey, return_code: int) -> None:
        if key.item_type == QueueItemType.TRAINING:
            await self._finalize_training_job(key.item_id, return_code)
        elif key.item_type == QueueItemType.SAMPLING:
            await self._finalize_sampling_run(key.item_id, return_code)

    async def _finalize_training_job(self, job_id: int, return_code: int) -> None:
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
            if return_code == 0 and job.status == JobStatus.COMPLETED:
                await self._enqueue_auto_sampling(job_id)

    async def _finalize_sampling_run(self, sampling_run_id: int, return_code: int) -> None:
        async with session_factory() as session:
            repo = SamplingRunRepository(session)
            sampling_run = await repo.get_by_id(sampling_run_id)
            if sampling_run is None:
                return
            if sampling_run.status == SamplingRunStatus.CANCELLED:
                logger.info(
                    "Sampling run id=%d finished after cancellation (exit code %d)",
                    sampling_run_id,
                    return_code,
                )
                await repo.clear_process_state(sampling_run)
                await session.commit()
                return
            if sampling_run.status in (SamplingRunStatus.RUNNING, SamplingRunStatus.QUEUED):
                if return_code == 0:
                    final_status = SamplingRunStatus.COMPLETED
                    await repo.update_status(sampling_run, final_status)
                else:
                    final_status = SamplingRunStatus.FAILED
                    error_message = (
                        sampling_run.error_message
                        or f"Sampling process exited with code {return_code}"
                    )
                    await repo.update_status(
                        sampling_run,
                        final_status,
                        error_message=error_message,
                    )
                await session.commit()
                logger.info(
                    "Sampling run id=%d finished with status=%s (exit code %d)",
                    sampling_run_id,
                    final_status,
                    return_code,
                )

    async def _enqueue_auto_sampling(self, job_id: int) -> None:
        async with session_factory() as session:
            job_repo = TrainingJobRepository(session)
            queue_repo = QueueRepository(session)
            sampling_run_repo = SamplingRunRepository(session)
            job = await job_repo.get_by_id(job_id)
            if job is None:
                return
            service = SamplingService(sampling_run_repo, queue_repo, job_repo)
            try:
                sampling_run = await service.create_auto_run_for_job(job)
            except SamplingCheckpointsNotFoundError:
                logger.warning("Post-training sampling requested for job id=%d, but no checkpoints were found", job_id)
                return
            if sampling_run is not None:
                await session.commit()
                logger.info("Queued post-training sampling run id=%d for job id=%d", sampling_run.id, job_id)

    async def _watch_cancellations(self) -> None:
        interval = settings.training.cancel_poll_interval_seconds
        while True:
            try:
                for key, managed in list(self._active_jobs.items()):
                    if not managed.is_running():
                        continue
                    if await self._get_item_cancelled(key) and managed.pid is not None:
                        logger.info(
                            "Cancellation requested for %s id=%d, killing pid=%d",
                            key.item_type,
                            key.item_id,
                            managed.pid,
                        )
                        self._kill_process_tree(managed.pid)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cancellation watcher error")
            await asyncio.sleep(interval)

    def _build_command(self, key: _QueueItemKey) -> list[str]:
        if key.item_type == QueueItemType.TRAINING:
            return [sys.executable, "-u", "-m", "src.trainer.runner", "--job-id", str(key.item_id)]
        if key.item_type == QueueItemType.SAMPLING:
            return [sys.executable, "-u", "-m", "src.sampler.runner", "--sampling-run-id", str(key.item_id)]
        raise ValueError(f"Unsupported queue item type: {key.item_type}")

    async def _run_entry(self, entry: QueueEntry) -> None:
        if entry.id is None:
            return
        key = _QueueItemKey(entry.item_type, entry.item_id)
        await self._dequeue_entry(entry.id)
        managed: _ManagedProcess | None = None
        try:
            logger.info("Spawning %s subprocess for id=%d", key.item_type, key.item_id)
            proc = subprocess.Popen(
                self._build_command(key),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            managed = _ManagedProcess(proc)
            self._active_jobs[key] = managed
            if managed.pid is not None:
                await self._mark_item_running(key, managed.pid)

            if self._echo_subprocess_output:
                await asyncio.gather(
                    asyncio.to_thread(_log_subprocess_output, proc, f"{key.item_type} {key.item_id}"),
                    managed.wait(),
                )
            else:
                await asyncio.gather(
                    asyncio.to_thread(_drain_subprocess_output, proc),
                    managed.wait(),
                )
        except Exception as exc:
            logger.exception("Failed to spawn %s for id=%d", key.item_type, key.item_id)
            await self._mark_item_spawn_failed(key, str(exc))
        finally:
            self._active_jobs.pop(key, None)
            if managed is not None:
                await self._finalize_item(key, managed.returncode or 0)

    async def _poll_loop(self) -> None:
        interval = settings.training.worker_poll_interval_seconds
        while True:
            try:
                if not await self._is_any_job_running():
                    entry = await self._get_next_queued_entry()
                    if entry is not None:
                        key = _QueueItemKey(entry.item_type, entry.item_id)
                        if key not in self._active_jobs:
                            task = asyncio.create_task(self._run_entry(entry))
                            self._job_tasks.add(task)
                            task.add_done_callback(self._job_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker poll error")
            await asyncio.sleep(interval)
