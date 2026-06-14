import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import patch

from src.db.migrations import migrate_schema
from src.db.tables.sampling_run import SamplingRun, SamplingRunStatus
from src.sampler.runner import _run
from src.settings.app_settings import settings


@pytest_asyncio.fixture
async def runner_db(tmp_path) -> tuple[AsyncSession, async_sessionmaker[AsyncSession], str]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
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
    sampling_run = SamplingRun(
        name="bad config",
        config_yaml="not_a_valid_config: [[[",
        lora_paths_yaml="[]\n",
        status=SamplingRunStatus.QUEUED,
    )
    session.add(sampling_run)
    await session.commit()
    await session.refresh(sampling_run)

    with patch("src.sampler.runner.session_factory", test_session_factory), patch.object(
        settings.training,
        "logs_dir",
        logs_dir,
    ), pytest.raises(SystemExit) as exc_info:
        await _run(sampling_run.id)

    assert exc_info.value.code == 1
    await session.refresh(sampling_run)
    assert sampling_run.status == SamplingRunStatus.FAILED
    assert sampling_run.error_message is not None
    assert sampling_run.log_path is not None
    from pathlib import Path

    log_path = Path(logs_dir) / f"sampling_run_{sampling_run.id}.log"
    assert log_path.exists()
    assert "failed" in log_path.read_text(encoding="utf-8").lower()
