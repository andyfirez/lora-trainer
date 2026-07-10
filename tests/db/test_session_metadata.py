import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.alembic_runner import run_migrations
from src.db.repositories.job_repo import JobRepository
from src.db.session import register_all_tables
from src.db.tables.job import Job, JobType


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session
    await engine.dispose()


@pytest.mark.asyncio
async def test_update_log_path_resolves_source_job_foreign_key(session: AsyncSession, tmp_path) -> None:
    training_job = Job(
        job_type=JobType.TRAINING,
        name="job",
        config_yaml="base_model_name: x",
    )
    session.add(training_job)
    await session.flush()
    sampling_job = Job(
        job_type=JobType.SAMPLING,
        name="sample",
        config_yaml="base_model_name: x",
        lora_paths_yaml="[]",
        source_job_id=training_job.id,
    )
    session.add(sampling_job)
    await session.commit()
    await session.refresh(sampling_job)

    repo = JobRepository(session)
    log_path = tmp_path / "sampling_job.log"
    await repo.update_log_path(sampling_job, str(log_path))
    await session.commit()

    assert sampling_job.log_path == str(log_path)


@pytest.mark.asyncio
async def test_session_factory_registers_all_tables_for_subprocess_metadata() -> None:
    register_all_tables()
    table_names = set(SQLModel.metadata.tables.keys())
    assert "job_configs" in table_names
    assert "jobs" in table_names


def test_run_migrations_applies_initial_schema(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.settings.app_settings.settings.database.path", str(db_path))
    run_migrations()
    assert db_path.is_file()

    import sqlite3

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    conn.close()
    assert tables == {
        "alembic_version",
        "datasets",
        "dataset_image_crops",
        "job_configs",
        "jobs",
        "queue_entries",
    }
