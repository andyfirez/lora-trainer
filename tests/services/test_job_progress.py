import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.migrations import migrate_schema
from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.training_job import JobStatus
from src.services.jobs.service import JobsService


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
async def test_update_progress_initial_step_zero(jobs_service: JobsService, session: AsyncSession) -> None:
    job = await jobs_service.create_job("test", "base_model_name: x")
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)

    updated = await jobs_service._job_repo.update_progress(
        job,
        step=0,
        total=500,
        loss=0.0,
        avr_loss=0.0,
        epoch=0,
        epoch_total=10,
    )
    await session.commit()

    assert updated.progress_step == 0
    assert updated.progress_total == 500
    assert updated.progress_loss == 0.0
    assert updated.progress_avr_loss == 0.0
    assert updated.progress_epoch == 0
    assert updated.progress_epoch_total == 10


@pytest.mark.asyncio
async def test_update_progress_after_training_step(jobs_service: JobsService, session: AsyncSession) -> None:
    job = await jobs_service.create_job("test", "base_model_name: x")
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    await jobs_service._job_repo.update_progress(
        job,
        step=0,
        total=500,
        loss=0.0,
        avr_loss=0.0,
        epoch=0,
        epoch_total=10,
    )

    updated = await jobs_service._job_repo.update_progress(
        job,
        step=42,
        total=500,
        loss=0.12,
        avr_loss=0.15,
        epoch=2,
        epoch_total=10,
    )
    await session.commit()

    assert updated.progress_step == 42
    assert updated.progress_total == 500
    assert updated.progress_loss == 0.12
    assert updated.progress_avr_loss == 0.15
    assert updated.progress_epoch == 2
