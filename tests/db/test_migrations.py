import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.migrations import migrate_schema
from src.db.repositories.queue_repo import QueueRepository
from src.db.tables.queue_entry import QueueEntry, QueueItemType


@pytest.mark.asyncio
async def test_queue_entry_migration_drops_legacy_job_id_column() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            """
            CREATE TABLE queue_entries (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                added_at DATETIME NOT NULL
            )
            """
        )
        await conn.exec_driver_sql(
            """
            INSERT INTO queue_entries (job_id, position, added_at)
            VALUES (7, 1, '2026-06-14 12:00:00')
            """
        )
        await migrate_schema(conn)

        columns = {
            row[1]
            for row in (await conn.exec_driver_sql("PRAGMA table_info(queue_entries)")).fetchall()
        }
        assert "job_id" not in columns
        assert {"item_type", "item_id", "position", "added_at"}.issubset(columns)

        rows = (await conn.exec_driver_sql("SELECT item_type, item_id FROM queue_entries")).fetchall()
        assert rows == [("training", 7)]

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        repo = QueueRepository(session)
        entry = await repo.add(
            QueueEntry(item_type=QueueItemType.SAMPLING, item_id=42, position=2),
        )
        await session.commit()
        assert entry.item_type == QueueItemType.SAMPLING
        assert entry.item_id == 42

    await engine.dispose()
