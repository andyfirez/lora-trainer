import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import _get_jobs_service
from src.api.main import app
from src.db.migrations import migrate_schema
from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.training_job import JobStatus, TrainingJob
from src.services.jobs.exceptions import JobNotCancellableError
from src.services.jobs.service import JobsService
from src.trainer.training_log import JobTrainingLogger


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session
    await engine.dispose()


@pytest_asyncio.fixture
async def jobs_service(session: AsyncSession) -> JobsService:
    return JobsService(TrainingJobRepository(session), QueueRepository(session))


@pytest.mark.asyncio
async def test_cancel_running_job(jobs_service: JobsService, session: AsyncSession) -> None:
    job = await jobs_service.create_job("test", "base_model_name: x")
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    cancelled = await jobs_service.cancel_job(job.id)
    assert cancelled.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_completed_job_raises(jobs_service: JobsService, session: AsyncSession) -> None:
    job = await jobs_service.create_job("test", "base_model_name: x")
    await jobs_service._job_repo.update_status(job, JobStatus.COMPLETED)
    with pytest.raises(JobNotCancellableError):
        await jobs_service.cancel_job(job.id)


@pytest.mark.asyncio
async def test_get_job_logs_tail(tmp_path, jobs_service: JobsService, session: AsyncSession) -> None:
    job = await jobs_service.create_job("test", "base_model_name: x")
    log_path = tmp_path / "job.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    job.log_path = str(log_path)
    session.add(job)
    await session.commit()
    lines = await jobs_service.get_job_logs(job.id, tail=2)
    assert lines == ["line2", "line3"]


@pytest.mark.asyncio
async def test_job_logs_api_endpoint(tmp_path) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as db_session:
        job = TrainingJob(name="api-test", config_yaml="base_model_name: x")
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)
        log_path = tmp_path / "job_api.log"
        log_path.write_text("hello\nworld\n", encoding="utf-8")
        job.log_path = str(log_path)
        db_session.add(job)
        await db_session.commit()

        async def _override_jobs_service():
            yield JobsService(TrainingJobRepository(db_session), QueueRepository(db_session))

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
