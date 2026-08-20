"""Subprocess runnable worker — polls loras/samplings and spawns child processes."""

import asyncio
import logging
import subprocess
from dataclasses import dataclass

import psutil
from src.db.repositories.runnable_queries import RunnableKind, get_by_kind, next_queued
from src.db.session import session_factory
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.runnable import runtime
from src.services.runnable.handlers import get_runnable_handler
from src.settings.app_settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingSpawn:
    """Placeholder in _active while a subprocess is being started."""

    def is_running(self) -> bool:
        return True

    @property
    def pid(self) -> None:
        return None


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


_ActiveEntry = _PendingSpawn | _ManagedProcess


def _drain_subprocess_output(proc: subprocess.Popen[bytes]) -> list[str]:
    if proc.stdout is None:
        return []
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line.decode(errors="replace").rstrip())
    return lines


def _log_subprocess_output(proc: subprocess.Popen[bytes], label: str) -> list[str]:
    if proc.stdout is None:
        return []
    lines: list[str] = []
    for line in proc.stdout:
        text = line.decode(errors="replace").rstrip()
        lines.append(text)
        logger.info("[%s] %s", label, text)
    return lines


def _summarize_subprocess_failure(lines: list[str], return_code: int, *, max_lines: int = 20) -> str:
    tail = [line for line in lines if line.strip()][-max_lines:]
    if tail:
        return f"Process exited with code {return_code}:\n" + "\n".join(tail)
    return f"Process exited with code {return_code}"


class SubprocessRunnableWorker:
    def __init__(self, *, echo_subprocess_output: bool = False) -> None:
        self._echo_subprocess_output = echo_subprocess_output
        self._active: dict[tuple[RunnableKind, int], _ActiveEntry] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._cancel_task: asyncio.Task[None] | None = None
        self._entry_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        logger.info(
            "Runnable worker started — polling every %ds, max %d concurrent job(s)",
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

        for (kind, entity_id), managed in list(self._active.items()):
            if isinstance(managed, _PendingSpawn):
                continue
            if managed.is_running() and managed.pid is not None:
                logger.info("Shutting down — terminating %s id=%d pid=%d", kind, entity_id, managed.pid)
                self._kill_process_tree(managed.pid)

        if self._entry_tasks:
            await asyncio.gather(*self._entry_tasks, return_exceptions=True)
        self._entry_tasks.clear()
        logger.info("Runnable worker stopped")

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

    async def _mark_running(self, kind: RunnableKind, entity_id: int, pid: int) -> None:
        async with session_factory() as session:
            entity = await get_by_kind(session, kind, entity_id)
            if entity is not None:
                runtime.mark_running(entity, pid=pid)
                session.add(entity)
            await session.commit()

    async def _mark_spawn_failed(self, kind: RunnableKind, entity_id: int, error_message: str) -> None:
        async with session_factory() as session:
            entity = await get_by_kind(session, kind, entity_id)
            if entity is not None and entity.status == RunnableStatus.QUEUED:
                runtime.mark_finished(entity, RunnableStatus.FAILED, error_message=error_message)
                session.add(entity)
            await session.commit()

    async def _is_cancelled(self, kind: RunnableKind, entity_id: int) -> bool:
        async with session_factory() as session:
            entity = await get_by_kind(session, kind, entity_id)
            return entity is not None and entity.status == RunnableStatus.CANCELLED

    async def _finalize(
        self,
        kind: RunnableKind,
        entity_id: int,
        return_code: int,
        output_lines: list[str] | None = None,
    ) -> None:
        async with session_factory() as session:
            entity = await get_by_kind(session, kind, entity_id)
            if entity is None:
                return
            if entity.status == RunnableStatus.CANCELLED:
                logger.info("%s id=%d finished after cancellation (exit code %d)", kind, entity_id, return_code)
                await session.commit()
                return
            error_message = None if return_code == 0 else _summarize_subprocess_failure(output_lines or [], return_code)
            await get_runnable_handler(kind).finalize(session, entity_id, return_code, error_message=error_message)
            await session.commit()
            logger.info("%s id=%d finished (exit code %d)", kind, entity_id, return_code)

    async def _watch_cancellations(self) -> None:
        interval = settings.training.cancel_poll_interval_seconds
        while True:
            try:
                for (kind, entity_id), managed in list(self._active.items()):
                    if isinstance(managed, _PendingSpawn):
                        continue
                    if not managed.is_running():
                        continue
                    if await self._is_cancelled(kind, entity_id) and managed.pid is not None:
                        logger.info("Cancellation requested for %s id=%d, killing pid=%d", kind, entity_id, managed.pid)
                        self._kill_process_tree(managed.pid)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cancellation watcher error")
            await asyncio.sleep(interval)

    async def _run_entity(self, kind: RunnableKind, entity_id: int) -> None:
        managed: _ManagedProcess | None = None
        output_lines: list[str] = []
        try:
            logger.info("Spawning %s subprocess for id=%d", kind, entity_id)
            command = get_runnable_handler(kind).build_command(entity_id)
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            managed = _ManagedProcess(proc)
            self._active[(kind, entity_id)] = managed
            if managed.pid is not None:
                await self._mark_running(kind, entity_id, managed.pid)

            if self._echo_subprocess_output:
                output_lines = await asyncio.to_thread(_log_subprocess_output, proc, f"{kind} {entity_id}")
                await managed.wait()
            else:
                output_lines, _ = await asyncio.gather(
                    asyncio.to_thread(_drain_subprocess_output, proc),
                    managed.wait(),
                )
        except Exception as exc:
            logger.exception("Failed to spawn %s for id=%d", kind, entity_id)
            await self._mark_spawn_failed(kind, entity_id, str(exc))
        finally:
            self._active.pop((kind, entity_id), None)
            if managed is not None:
                await self._finalize(kind, entity_id, managed.returncode or 0, output_lines)

    async def _active_count(self) -> int:
        return sum(1 for managed in self._active.values() if managed.is_running())

    async def _poll_loop(self) -> None:
        while True:
            try:
                max_jobs = settings.training.max_concurrent_jobs
                while await self._active_count() < max_jobs:
                    async with session_factory() as session:
                        peeked = await next_queued(session)
                    if peeked is None or peeked in self._active:
                        break
                    kind, entity_id = peeked
                    self._active[(kind, entity_id)] = _PendingSpawn()
                    async with session_factory() as session:
                        entity = await get_by_kind(session, kind, entity_id)
                        if entity is None or entity.status != RunnableStatus.QUEUED:
                            self._active.pop((kind, entity_id), None)
                            break
                        runtime.remove_from_queue(entity)
                        session.add(entity)
                        await session.commit()
                    task = asyncio.create_task(self._run_entity(kind, entity_id))
                    self._entry_tasks.add(task)
                    task.add_done_callback(self._entry_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker poll error")
            await asyncio.sleep(settings.training.worker_poll_interval_seconds)
