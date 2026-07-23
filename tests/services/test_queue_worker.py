import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.db.tables.dataset import Dataset
from src.db.tables.job import Job, JobStatus, JobType
from src.db.tables.job_config import ConfigType
from src.db.tables.queue_entry import QueueEntry
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService
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


@asynccontextmanager
async def _session_with_job(job_id: int, job_type: JobType = JobType.TRAINING):
    job = Job(id=job_id, job_type=job_type, name="test", config_yaml="base_model_name: x")
    session = AsyncMock()
    repo = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=job)

    @asynccontextmanager
    async def factory():
        yield session

    with patch("src.services.worker.service.session_factory", factory), patch(
        "src.services.worker.service.JobRepository",
        return_value=repo,
    ):
        yield


@pytest.mark.asyncio
async def test_queue_worker_start_stop() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    with patch.object(worker, "_get_next_queued_entry", AsyncMock(return_value=None)):
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
        async with _session_with_job(7):
            with patch(
                "src.services.worker.service.subprocess.Popen",
                MagicMock(return_value=fake_proc),
            ), patch.object(worker, "_dequeue_entry", AsyncMock()), patch.object(
                worker, "_mark_job_running", AsyncMock()
            ), patch.object(worker, "_finalize_job", AsyncMock()):
                await worker._run_entry(QueueEntry(id=1, job_id=7, position=1))

    assert not any("[training 7]" in record.message for record in caplog.records)


async def _inline_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_echo_subprocess_output_true_logs_subprocess_lines() -> None:
    worker = QueueWorker(echo_subprocess_output=True)
    fake_proc = _FakePopen([b"visible line\n"])

    async with _session_with_job(9):
        with patch(
            "src.services.worker.service.asyncio.to_thread",
            side_effect=_inline_to_thread,
        ), patch(
            "src.services.worker.service._log_subprocess_output",
        ) as log_mock, patch(
            "src.services.worker.service.subprocess.Popen",
            MagicMock(return_value=fake_proc),
        ), patch.object(worker, "_dequeue_entry", AsyncMock()), patch.object(
            worker, "_mark_job_running", AsyncMock()
        ), patch.object(worker, "_finalize_job", AsyncMock()):
            await worker._run_entry(QueueEntry(id=1, job_id=9, position=1))

    log_mock.assert_called_once_with(fake_proc, "training 9")


@pytest.mark.asyncio
async def test_run_job_dequeues_before_spawn() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    fake_proc = _FakePopen([])
    call_order: list[str] = []

    async def _dequeue(entry_id: int) -> None:
        call_order.append("dequeue")

    def _popen(*_args: object, **_kwargs: object) -> _FakePopen:
        call_order.append("popen")
        return fake_proc

    async with _session_with_job(3):
        with patch(
            "src.services.worker.service.subprocess.Popen",
            side_effect=_popen,
        ), patch.object(worker, "_dequeue_entry", side_effect=_dequeue), patch.object(
            worker, "_mark_job_running", AsyncMock()
        ), patch.object(worker, "_finalize_job", AsyncMock()):
            await worker._run_entry(QueueEntry(id=5, job_id=3, position=1))

    assert call_order == ["dequeue", "popen"]


@pytest.mark.asyncio
async def test_finalize_job_does_not_dequeue(
    worker_db: tuple[AsyncSession, async_sessionmaker[AsyncSession]],
) -> None:
    session, test_session_factory = worker_db
    job = Job(
        job_type=JobType.TRAINING,
        name="test",
        config_yaml="base_model_name: x",
        status=JobStatus.RUNNING,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    worker = QueueWorker()

    with patch("src.services.worker.service.session_factory", test_session_factory), patch.object(
        worker, "_dequeue_entry", AsyncMock()
    ) as dequeue_mock:
        await worker._finalize_job(job.id, 0)

    dequeue_mock.assert_not_called()


@pytest.mark.asyncio
async def test_poll_loop_skips_active_job() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    active = MagicMock()
    active.is_running.return_value = True
    worker._active_jobs[42] = active

    with patch.object(
        worker, "_get_next_queued_entry", AsyncMock(
            return_value=QueueEntry(id=1, job_id=42, position=1)
        )
    ), patch.object(worker, "_run_entry", AsyncMock()) as run_job_mock:
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

    async with _session_with_job(11):
        with patch.object(worker, "_dequeue_entry", AsyncMock()), patch(
            "src.services.worker.service.subprocess.Popen",
            side_effect=OSError("spawn failed"),
        ), patch.object(worker, "_mark_job_spawn_failed", AsyncMock()) as mark_failed_mock, patch.object(
            worker, "_finalize_job", AsyncMock()
        ) as finalize_mock:
            await worker._run_entry(QueueEntry(id=1, job_id=11, position=1))

    mark_failed_mock.assert_awaited_once_with(11, "spawn failed")
    finalize_mock.assert_not_called()


@pytest_asyncio.fixture
async def worker_db() -> tuple[AsyncSession, async_sessionmaker[AsyncSession]]:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session, factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_dequeue_on_start_prevents_requeue_after_completion(
    worker_db: tuple[AsyncSession, async_sessionmaker[AsyncSession]],
) -> None:
    session, test_session_factory = worker_db
    queue_repo = QueueRepository(session)
    job = Job(
        job_type=JobType.TRAINING,
        name="test",
        config_yaml="base_model_name: x",
        status=JobStatus.QUEUED,
    )
    session.add(job)
    await session.flush()
    entry = await queue_repo.add(QueueEntry(job_id=job.id, position=1))
    await session.commit()
    job_id = job.id

    worker = QueueWorker()

    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._dequeue_entry(entry.id)
        async with test_session_factory() as db_session:
            repo = JobRepository(db_session)
            stored_job = await repo.get_by_id(job_id)
            assert stored_job is not None
            await repo.update_status(stored_job, JobStatus.COMPLETED)
            await db_session.commit()
        assert await worker._get_next_queued_entry() is None
        assert await worker._is_any_job_running() is False


@pytest.mark.asyncio
async def test_build_command_dispatches_sampling_runner() -> None:
    worker = QueueWorker()

    command = worker._build_command(12, JobType.SAMPLING)

    assert command[-3:] == ["src.sampler.runner", "--job-id", "12"]


@pytest.mark.asyncio
async def test_finalize_job_marks_sampling_completed(
    worker_db: tuple[AsyncSession, async_sessionmaker[AsyncSession]],
) -> None:
    session, test_session_factory = worker_db
    sampling_job = Job(
        job_type=JobType.SAMPLING,
        name="sample",
        config_yaml="base_model_name: x",
        lora_paths_yaml="[]",
        status=JobStatus.RUNNING,
    )
    session.add(sampling_job)
    await session.commit()
    await session.refresh(sampling_job)

    worker = QueueWorker()
    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._finalize_job(sampling_job.id, 0)

    await session.refresh(sampling_job)
    assert sampling_job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_finalize_job_preserves_error_message(
    worker_db: tuple[AsyncSession, async_sessionmaker[AsyncSession]],
) -> None:
    session, test_session_factory = worker_db
    sampling_job = Job(
        job_type=JobType.SAMPLING,
        name="sample",
        config_yaml="base_model_name: x",
        lora_paths_yaml="[]",
        status=JobStatus.RUNNING,
        error_message="CUDA is not available",
    )
    session.add(sampling_job)
    await session.commit()
    await session.refresh(sampling_job)

    worker = QueueWorker()
    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._finalize_job(sampling_job.id, 1)

    await session.refresh(sampling_job)
    assert sampling_job.status == JobStatus.FAILED
    assert sampling_job.error_message == "CUDA is not available"


@pytest.mark.asyncio
async def test_finalize_job_includes_subprocess_output_on_failure(
    worker_db: tuple[AsyncSession, async_sessionmaker[AsyncSession]],
) -> None:
    session, test_session_factory = worker_db
    job = Job(
        job_type=JobType.TRAINING,
        name="test",
        config_yaml="base_model_name: x",
        status=JobStatus.RUNNING,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    worker = QueueWorker()
    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._finalize_job(
            job.id,
            1,
            ["Traceback (most recent call last):", "NameError: something broke"],
        )

    await session.refresh(job)
    assert job.status == JobStatus.FAILED
    assert "Process exited with code 1" in job.error_message
    assert "NameError: something broke" in job.error_message


@pytest.mark.asyncio
async def test_finalize_job_queues_sampling_when_runner_already_completed(
    session: AsyncSession,
    jobs_service: JobsService,
    config_service: JobConfigService,
    training_dataset: Dataset,
    storage_roots,
) -> None:
    output_rel = "output"

    sampling_config = await config_service.create_config(
        name="post-train sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - test prompt\n",
    )
    training_config = await config_service.create_config(
        name="training",
        config_type=ConfigType.TRAINING,
        config_yaml=f"""
base_model_name: test-model
output_dir: {output_rel}
lora_name: demo
output_format: safetensors
checkpointing_enabled: true
sampling_enabled: true
sampling_config_id: {sampling_config.id}
concepts:
  - dataset_id: {training_dataset.id}
""",
    )
    training_job = await jobs_service.create_from_config(training_config.id)
    from src.trainer.config import TrainConfig

    train_config = TrainConfig.from_yaml(training_job.config_yaml)
    work_dir = storage_roots["lora"] / output_rel / train_config.lora_name
    work_dir.mkdir(parents=True)
    (work_dir / f"{train_config.lora_name}_epoch1.safetensors").write_bytes(b"epoch")
    await jobs_service._job_repo.update_status(training_job, JobStatus.COMPLETED)
    await session.commit()

    @asynccontextmanager
    async def test_session_factory():
        yield session

    worker = QueueWorker()
    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._finalize_job(training_job.id, 0)

    sampling_jobs = await jobs_service.list_jobs_by_source(training_job.id)
    assert len(sampling_jobs) == 1
    assert sampling_jobs[0].job_type == JobType.SAMPLING
    assert sampling_jobs[0].status == JobStatus.QUEUED
    queue_entry = await jobs_service._queue_repo.get_by_job_id(sampling_jobs[0].id)
    assert queue_entry is not None


@pytest.mark.asyncio
async def test_finalize_job_registers_trained_lora_without_sampling(
    session: AsyncSession,
    jobs_service: JobsService,
    config_service: JobConfigService,
    training_dataset: Dataset,
    storage_roots,
) -> None:
    output_rel = "output"
    training_config = await config_service.create_config(
        name="training",
        config_type=ConfigType.TRAINING,
        config_yaml=f"""
base_model_name: test-model
output_dir: {output_rel}
lora_name: demo
output_format: safetensors
sampling_enabled: false
concepts:
  - dataset_id: {training_dataset.id}
""",
    )
    training_job = await jobs_service.create_from_config(training_config.id)
    from src.db.repositories.trained_lora_repo import TrainedLoraRepository
    from src.trainer.config import TrainConfig

    train_config = TrainConfig.from_yaml(training_job.config_yaml)
    work_dir = storage_roots["lora"] / output_rel / train_config.lora_name
    work_dir.mkdir(parents=True)
    (work_dir / f"{train_config.lora_name}.safetensors").write_bytes(b"weights")
    await jobs_service._job_repo.update_output_path(training_job, str(work_dir))
    await jobs_service._job_repo.update_status(training_job, JobStatus.COMPLETED)
    await session.commit()

    @asynccontextmanager
    async def test_session_factory():
        yield session

    worker = QueueWorker()
    with patch("src.services.worker.service.session_factory", test_session_factory):
        await worker._finalize_job(training_job.id, 0)

    loras = await TrainedLoraRepository(session).list_all()
    assert len(loras) == 1
    assert loras[0].job_id == training_job.id
    assert loras[0].name == train_config.lora_name
    assert loras[0].weights_relpath.endswith(f"{train_config.lora_name}.safetensors")
    sampling_jobs = await jobs_service.list_jobs_by_source(training_job.id)
    assert sampling_jobs == []
