from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.session import register_all_tables
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.sampler.job_runner import run_sampling
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
async def test_run_invalid_config_writes_log_and_returns_failure_exit_code(
    runner_db: tuple[AsyncSession, async_sessionmaker[AsyncSession], str],
) -> None:
    session, test_session_factory, logs_dir = runner_db
    sampling = Sampling(
        name="bad config",
        config={"parameters": "not-a-dict"},
        lora_paths=[],
        status=RunnableStatus.RUNNING,
    )
    session.add(sampling)
    await session.commit()
    await session.refresh(sampling)

    with (
        patch("src.services.runnable.db_updates.session_factory", test_session_factory),
        patch.object(settings.training, "logs_dir", logs_dir),
    ):
        exit_code = await run_sampling(sampling.id)

    assert exit_code == 1
    log_path = Path(logs_dir) / f"sampling_{sampling.id}.log"
    assert log_path.exists()
    assert "failed" in log_path.read_text(encoding="utf-8").lower()

    await session.refresh(sampling)
    assert sampling.log_path == str(log_path)
