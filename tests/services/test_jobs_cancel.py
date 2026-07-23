import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.api.dependencies import _get_jobs_service
from src.api.main import app
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.db.tables.job import Job, JobStatus, JobType
from src.services.jobs.exceptions import JobNotCancellableError
from src.services.jobs.service import JobsService
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import MetricLogger
from src.trainer.sdxl.checkpoint_state import save_resume_state
from src.trainer.training_log import JobTrainingLogger


@pytest.mark.asyncio
async def test_cancel_running_job(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    cancelled = await jobs_service.cancel_job(job.id)
    assert cancelled.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_clears_progress_and_error(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(
        job,
        JobStatus.RUNNING,
        pid=1234,
        error_message="previous failure",
    )
    await jobs_service._job_repo.update_progress(
        job,
        step=50,
        total=100,
        loss=0.5,
        avr_loss=0.4,
        epoch=1,
        epoch_total=10,
    )
    await jobs_service._job_repo.update_cache_progress(job, step=3, total=10)

    cancelled = await jobs_service.cancel_job(job.id)

    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.pid is None
    assert cancelled.error_message == "previous failure"
    assert cancelled.progress_step == 50
    assert cancelled.progress_total == 100
    assert cancelled.progress_loss == 0.5
    assert cancelled.progress_avr_loss == 0.4
    assert cancelled.progress_epoch == 1
    assert cancelled.progress_epoch_total == 10
    assert cancelled.cache_progress_step == 3
    assert cancelled.cache_progress_total == 10


@pytest.mark.asyncio
async def test_enqueue_clears_stale_runtime_state(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(
        job,
        JobStatus.FAILED,
        error_message="CUDA OOM",
    )
    await jobs_service._job_repo.update_progress(job, step=25, total=100, loss=0.9, avr_loss=0.8)

    queued = await jobs_service.enqueue_job(job.id)
    refreshed = await jobs_service.get_job(job.id)

    assert queued.job_id == job.id
    assert refreshed.status == JobStatus.QUEUED
    assert refreshed.error_message is None
    assert refreshed.progress_step is None
    assert refreshed.progress_total is None
    assert refreshed.progress_loss is None
    assert refreshed.progress_avr_loss is None


@pytest.mark.asyncio
async def test_resume_job_queues_with_resume_state(
    jobs_service: JobsService,
    create_training_job,
    training_dataset,
    tmp_path,
) -> None:
    output_dir = tmp_path / "output"
    config_yaml = f"""
output_dir: {output_dir.as_posix()}
lora_name: test_lora
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - dataset_id: {training_dataset.id}
"""
    job = await create_training_job(config_yaml=config_yaml)
    train_config = TrainConfig.from_yaml(job.config_yaml)
    work_dir = output_dir / train_config.lora_name
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / f"{train_config.lora_name}_epoch3.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    save_resume_state(
        checkpoint_path=checkpoint_path,
        lora_state_dict={},
        optimizer_state_dict={},
        lr_scheduler_state_dict={},
        epoch=3,
        global_step=123,
        epoch_step=0,
    )
    await jobs_service._job_repo.update_status(job, JobStatus.FAILED, error_message="boom")

    await jobs_service.resume_job(job.id)
    resumed = await jobs_service.get_job(job.id)

    assert resumed.status == JobStatus.QUEUED
    assert resumed.resume_checkpoint_path == str(checkpoint_path)
    assert resumed.resume_from_epoch == 3
    assert resumed.resume_from_step == 123


@pytest.mark.asyncio
async def test_cancel_running_job_with_save_sets_request_flag(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    await jobs_service._job_repo.update_progress(job, 10, 100)

    result = await jobs_service.cancel_job(job.id, save_checkpoint=True)

    assert result.status == JobStatus.RUNNING
    assert result.save_checkpoint_requested is True


@pytest.mark.asyncio
async def test_cancel_with_save_during_cache_immediate_cancel(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)

    result = await jobs_service.cancel_job(job.id, save_checkpoint=True)

    assert result.status == JobStatus.CANCELLED
    assert result.save_checkpoint_requested is False
    assert result.pid is None


@pytest.mark.asyncio
async def test_enqueue_clears_loss_log(
    jobs_service: JobsService,
    create_training_job,
    training_dataset,
    tmp_path,
) -> None:
    output_dir = tmp_path / "output"
    config_yaml = f"""
output_dir: {output_dir.as_posix()}
lora_name: test_lora
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - dataset_id: {training_dataset.id}
"""
    job = await create_training_job(config_yaml=config_yaml)
    train_config = TrainConfig.from_yaml(job.config_yaml)
    loss_log = output_dir / train_config.lora_name / "loss_log.db"
    logger = MetricLogger(loss_log)
    logger.log({"loss/loss": 0.9})
    logger.commit(step=10)
    logger.finish()

    await jobs_service.enqueue_job(job.id)

    assert not loss_log.exists()
    loss = await jobs_service.get_job_loss(job.id)
    assert loss.points == []


@pytest.mark.asyncio
async def test_cancel_completed_job_raises(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.COMPLETED)
    with pytest.raises(JobNotCancellableError):
        await jobs_service.cancel_job(job.id)


@pytest.mark.asyncio
async def test_get_job_logs_tail(
    tmp_path,
    jobs_service: JobsService,
    create_training_job,
    session: AsyncSession,
) -> None:
    job = await create_training_job()
    log_path = tmp_path / "job.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    job.log_path = str(log_path)
    session.add(job)
    await session.commit()
    lines = await jobs_service.get_job_logs(job.id, tail=2)
    assert lines == ["line2", "line3"]


@pytest.mark.asyncio
async def test_job_logs_api_endpoint(tmp_path) -> None:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as db_session:
        job = Job(
            job_type=JobType.TRAINING,
            name="api-test",
            config_yaml="base_model_name: x",
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)
        log_path = tmp_path / "job_api.log"
        log_path.write_text("hello\nworld\n", encoding="utf-8")
        job.log_path = str(log_path)
        db_session.add(job)
        await db_session.commit()

        async def _override_jobs_service():
            yield JobsService(
                JobRepository(db_session),
                QueueRepository(db_session),
                JobConfigRepository(db_session),
                DatasetRepository(db_session),
            )

        app.dependency_overrides[_get_jobs_service] = _override_jobs_service
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(f"/jobs/{job.id}/logs?tail=10")
                assert response.status_code == 200
                assert response.json()["lines"] == ["hello", "world"]
        finally:
            app.dependency_overrides.clear()

    await engine.dispose()


def test_read_tail_from_file(tmp_path) -> None:
    log_path = tmp_path / "tail.log"
    log_path.write_text("\n".join(f"line{i}" for i in range(10)), encoding="utf-8")
    lines = JobTrainingLogger.read_tail(log_path, lines=3)
    assert lines == ["line7", "line8", "line9"]


def test_job_logger_overwrites_log_file(tmp_path) -> None:
    log_path = tmp_path / "job.log"
    log_path.write_text("old content\n", encoding="utf-8")

    first_logger = JobTrainingLogger(job_id=1, log_path=log_path)
    first_logger.logger.info("first run")

    second_logger = JobTrainingLogger(job_id=2, log_path=log_path)
    second_logger.logger.info("second run")

    content = log_path.read_text(encoding="utf-8")
    assert "old content" not in content
    assert "first run" not in content
    assert "second run" in content
