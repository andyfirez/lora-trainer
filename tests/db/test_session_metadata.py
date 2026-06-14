import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.migrations import migrate_schema
from src.db.repositories.sampling_run_repo import SamplingRunRepository
from src.db.session import register_all_tables, session_factory
from src.db.tables.sampling_run import SamplingRun
from src.db.tables.training_job import TrainingJob


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session
    await engine.dispose()


@pytest.mark.asyncio
async def test_update_log_path_resolves_source_job_foreign_key(session: AsyncSession, tmp_path) -> None:
    job = TrainingJob(name="job", config_yaml="base_model_name: x")
    session.add(job)
    await session.flush()
    sampling_run = SamplingRun(
        name="sample",
        config_yaml="base_model_name: x",
        lora_paths_yaml="[]",
        source_job_id=job.id,
    )
    session.add(sampling_run)
    await session.commit()
    await session.refresh(sampling_run)

    repo = SamplingRunRepository(session)
    log_path = tmp_path / "sampling_run.log"
    await repo.update_log_path(sampling_run, str(log_path))
    await session.commit()

    assert sampling_run.log_path == str(log_path)


@pytest.mark.asyncio
async def test_session_factory_registers_all_tables_for_subprocess_metadata() -> None:
    register_all_tables()
    table_names = set(SQLModel.metadata.tables.keys())
    assert "training_jobs" in table_names
    assert "sampling_runs" in table_names
