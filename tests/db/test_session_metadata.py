import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.alembic_runner import run_migrations
from src.db.repositories.lora_repo import LoraRepository
from src.db.session import register_all_tables
from src.db.tables.lora import Lora


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
async def test_lora_repo_round_trips_fields(session: AsyncSession, tmp_path) -> None:
    lora = Lora(name="test-lora", base_model_name="test-model", config_yaml="base_model_name: x")
    session.add(lora)
    await session.commit()
    await session.refresh(lora)

    repo = LoraRepository(session)
    lora.log_path = str(tmp_path / "lora.log")
    session.add(lora)
    await session.commit()

    fetched = await repo.get_by_name("test-lora")
    assert fetched is not None
    assert fetched.log_path == str(tmp_path / "lora.log")


@pytest.mark.asyncio
async def test_session_factory_registers_all_tables_for_subprocess_metadata() -> None:
    register_all_tables()
    table_names = set(SQLModel.metadata.tables.keys())
    assert "loras" in table_names
    assert "samplings" in table_names


def test_run_migrations_applies_full_schema(tmp_path, monkeypatch) -> None:
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
        "loras",
        "samplings",
    }
