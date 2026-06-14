import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import patch

from src.db.session import register_all_tables
from src.db.tables.job import Job, JobStatus, JobType
from src.sampler.job_runner import run_sampling_job
from src.settings.app_settings import settings


@pytest_asyncio.fixture
async def runner_db(tmp_path) -> tuple[AsyncSession, async_sessionmaker[AsyncSession], str]:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    logs_dir = tmp_path / "logs"
    async with factory() as db_session:
        yield db_session, factory, str(logs_dir)
    await engine.dispose()


@pytest.mark.asyncio
async def test_run_invalid_config_writes_log_and_marks_failed(
    runner_db: tuple[AsyncSession, async_sessionmaker[AsyncSession], str],
) -> None:
    session, test_session_factory, logs_dir = runner_db
    sampling_job = Job(
        job_type=JobType.SAMPLING,
        name="bad config",
        config_yaml="not_a_valid_config: [[[",
        lora_paths_yaml="[]\n",
        status=JobStatus.QUEUED,
    )
    session.add(sampling_job)
    await session.commit()
    await session.refresh(sampling_job)

    with patch("src.sampler.job_runner.session_factory", test_session_factory), patch.object(
        settings.training,
        "logs_dir",
        logs_dir,
    ):
        exit_code = await run_sampling_job(sampling_job.id)

    assert exit_code == 1
    await session.refresh(sampling_job)
    assert sampling_job.status == JobStatus.FAILED
    assert sampling_job.error_message is not None
    assert sampling_job.log_path is not None
    from pathlib import Path

    log_path = Path(logs_dir) / f"job_{sampling_job.id}.log"
    assert log_path.exists()
    assert "failed" in log_path.read_text(encoding="utf-8").lower()
