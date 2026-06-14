import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.migrations import migrate_schema
from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.queue_entry import QueueEntry
from src.db.tables.training_job import JobStatus, TrainingJob
from src.services.worker.service import QueueWorker


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self._index = 0

    def __iter__(self) -> "_FakeStdout":
        return self

    def __next__(self) -> bytes:
        if self._index >= len(self._lines):
            raise StopIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class _FakePopen:
    def __init__(self, lines: list[bytes]) -> None:
        self.stdout = _FakeStdout(lines)
        self.pid = 4242
        self._returncode: int | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode

    def wait(self) -> int:
        self._returncode = 0
        return 0


@pytest.mark.asyncio
async def test_queue_worker_start_stop() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    with patch.object(worker, "_is_any_job_running", AsyncMock(return_value=False)), patch.object(
        worker, "_get_next_queued_job_id", AsyncMock(return_value=None)
    ):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()


@pytest.mark.asyncio
async def test_echo_subprocess_output_false_drains_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    fake_proc = _FakePopen([b"secret line\n"])

    with caplog.at_level(logging.INFO, logger="src.services.worker.service"):
        with patch(
            "src.services.worker.service.subprocess.Popen",
            MagicMock(return_value=fake_proc),
        ), patch.object(worker, "_dequeue_job", AsyncMock()), patch.object(
            worker, "_mark_job_running", AsyncMock()
        ), patch.object(worker, "_finalize_job", AsyncMock()):
            await worker._run_job(7)

    assert not any("[job 7]" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_echo_subprocess_output_true_logs_subprocess_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = QueueWorker(echo_subprocess_output=True)
    fake_proc = _FakePopen([b"visible line\n"])

    with caplog.at_level(logging.INFO, logger="src.services.worker.service"):
        with patch(
            "src.services.worker.service.subprocess.Popen",
            MagicMock(return_value=fake_proc),
        ), patch.object(worker, "_dequeue_job", AsyncMock()), patch.object(
            worker, "_mark_job_running", AsyncMock()
        ), patch.object(worker, "_finalize_job", AsyncMock()):
            await worker._run_job(9)

    assert any("[job 9] visible line" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_run_job_dequeues_before_spawn() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    fake_proc = _FakePopen([])
    call_order: list[str] = []

    async def _dequeue(job_id: int) -> None:
        call_order.append("dequeue")

    def _popen(*_args: object, **_kwargs: object) -> _FakePopen:
        call_order.append("popen")
        return fake_proc

    with patch(
        "src.services.worker.service.subprocess.Popen",
        side_effect=_popen,
    ), patch.object(worker, "_dequeue_job", side_effect=_dequeue), patch.object(
        worker, "_mark_job_running", AsyncMock()
    ), patch.object(worker, "_finalize_job", AsyncMock()):
        await worker._run_job(3)

    assert call_order == ["dequeue", "popen"]


@pytest.mark.asyncio
async def test_finalize_job_does_not_dequeue() -> None:
    worker = QueueWorker(echo_subprocess_output=False)

    with patch.object(worker, "_dequeue_job", AsyncMock()) as dequeue_mock:
        await worker._finalize_job(5, 0)

    dequeue_mock.assert_not_called()


@pytest.mark.asyncio
async def test_poll_loop_skips_active_job() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    worker._active_jobs[42] = MagicMock()

    with patch.object(worker, "_is_any_job_running", AsyncMock(return_value=False)), patch.object(
        worker, "_get_next_queued_job_id", AsyncMock(return_value=42)
    ), patch.object(worker, "_run_job", AsyncMock()) as run_job_mock:
        poll_task = asyncio.create_task(worker._poll_loop())
        await asyncio.sleep(0.05)
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

    run_job_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_job_spawn_failure_marks_failed() -> None:
    worker = QueueWorker(echo_subprocess_output=False)

    with patch.object(worker, "_dequeue_job", AsyncMock()), patch(
        "src.services.worker.service.subprocess.Popen",
        side_effect=OSError("spawn failed"),
    ), patch.object(worker, "_mark_job_spawn_failed", AsyncMock()) as mark_failed_mock, patch.object(
        worker, "_finalize_job", AsyncMock()
    ) as finalize_mock:
        await worker._run_job(11)

    mark_failed_mock.assert_awaited_once_with(11, "spawn failed")
    finalize_mock.assert_not_called()


@pytest_asyncio.fixture
async def worker_db() -> tuple[AsyncSession, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session, factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_dequeue_on_start_prevents_requeue_after_completion(
    worker_db: tuple[AsyncSession, async_sessionmaker[AsyncSession]],
) -> None:
    session, test_session_factory = worker_db
    job_repo = TrainingJobRepository(session)
    queue_repo = QueueRepository(session)
    job = TrainingJob(name="test", config_yaml="base_model_name: x", status=JobStatus.QUEUED)
    session.add(job)
    await session.flush()
    await queue_repo.add(QueueEntry(job_id=job.id, position=1))
    await session.commit()
    job_id = job.id

    worker = QueueWorker()

    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._dequeue_job(job_id)
        async with test_session_factory() as db_session:
            repo = TrainingJobRepository(db_session)
            stored_job = await repo.get_by_id(job_id)
            assert stored_job is not None
            await repo.update_status(stored_job, JobStatus.COMPLETED)
            await db_session.commit()
        assert await worker._get_next_queued_job_id() is None
        assert await worker._is_any_job_running() is False
