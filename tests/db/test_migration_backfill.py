"""Migration tests: legacy jobs/trained_loras schema upgraded through 016-019.

Seeds a DB at revision 015 (pre-refactor schema) with representative rows, then
upgrades to head and asserts the 018 backfill and 019 legacy-table drop behave
as documented in the plan.
"""

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config


def _alembic_config(db_path: Path) -> Config:
    project_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return cfg


@pytest.fixture(autouse=True)
def _isolated_lora_root(tmp_path, monkeypatch):
    """Revision 013 discovers LoRA work dirs from `[storage].lora_root` in config.toml.

    Point it at an empty directory so the real dev machine's LoRA output isn't
    picked up by these migration tests.
    """
    lora_root = tmp_path / "isolated-lora-root"
    lora_root.mkdir()
    config_path = tmp_path / "isolated-config.toml"
    config_path.write_text(f'[storage]\nlora_root = "{lora_root.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_path))


def _seed_legacy_rows(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    now = "2026-01-01T00:00:00+00:00"

    # A completed training job with a linked trained_loras row.
    conn.execute(
        "INSERT INTO jobs (id, job_type, name, status, config_yaml, output_path, log_path, "
        "accumulated_elapsed_seconds, save_checkpoint_requested, created_at, updated_at) "
        "VALUES (1, 'training', 'completed-job', 'completed', 'base_model_name: sd15', "
        "'/out/completed-lora', '/out/completed-lora/log.txt', 42.0, 0, ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO trained_loras (id, name, relative_path, weights_relpath, job_id, config_yaml, "
        "base_model_name, created_at, updated_at) "
        "VALUES (1, 'completed-lora', 'completed-lora', 'completed-lora/completed-lora.safetensors', 1, "
        "'base_model_name: sd15', 'sd15', ?, ?)",
        (now, now),
    )

    # A training job that was mid-flight when the app stopped -> should become orphan.
    conn.execute(
        "INSERT INTO jobs (id, job_type, name, status, config_yaml, accumulated_elapsed_seconds, "
        "save_checkpoint_requested, created_at, updated_at) "
        "VALUES (2, 'training', 'running-job', 'running', 'base_model_name: sdxl', 12.5, 0, ?, ?)",
        (now, now),
    )

    # A failed training job with no trained_loras row.
    conn.execute(
        "INSERT INTO jobs (id, job_type, name, status, config_yaml, error_message, "
        "accumulated_elapsed_seconds, save_checkpoint_requested, created_at, updated_at) "
        "VALUES (3, 'training', 'failed-job', 'failed', 'base_model_name: sd15', 'boom', 3.0, 0, ?, ?)",
        (now, now),
    )

    # A completed sampling job.
    conn.execute(
        "INSERT INTO jobs (id, job_type, name, status, config_yaml, lora_paths_yaml, "
        "accumulated_elapsed_seconds, save_checkpoint_requested, created_at, updated_at) "
        "VALUES (4, 'sampling', 'sampling-job', 'completed', ?, ?, 5.0, 0, ?, ?)",
        (
            "output_dir: /out/sampling\n",
            "- completed-lora/completed-lora.safetensors\n",
            now,
            now,
        ),
    )

    conn.commit()
    conn.close()


def test_upgrade_head_on_empty_db_succeeds(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    command.upgrade(_alembic_config(db_path), "head")
    assert db_path.is_file()


def test_backfill_migrates_legacy_jobs_into_loras_and_samplings(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "015")
    _seed_legacy_rows(db_path)
    command.upgrade(cfg, "head")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    assert tables == {"alembic_version", "datasets", "dataset_image_crops", "loras", "samplings"}

    loras = {row["name"]: row for row in conn.execute("SELECT * FROM loras").fetchall()}
    assert set(loras) == {"completed-lora", "running-job", "failed-job"}

    completed = loras["completed-lora"]
    assert completed["status"] == "completed"
    assert completed["relative_path"] == "completed-lora"
    assert completed["weights_relpath"] == "completed-lora/completed-lora.safetensors"
    assert completed["accumulated_elapsed_seconds"] == 42.0

    orphaned = loras["running-job"]
    assert orphaned["status"] == "orphan"

    failed = loras["failed-job"]
    assert failed["status"] == "failed"
    assert failed["error_message"] == "boom"

    samplings = {row["name"]: row for row in conn.execute("SELECT * FROM samplings").fetchall()}
    assert set(samplings) == {"sampling-job"}
    sampling = samplings["sampling-job"]
    assert sampling["status"] == "completed"
    assert sampling["lora_paths_yaml"] == "- completed-lora/completed-lora.safetensors\n"

    conn.close()


@pytest.mark.asyncio
async def test_backfilled_lora_status_reads_through_orm(tmp_path) -> None:
    """Regression: migrated rows store lowercase status values; ORM must read them."""
    db_path = tmp_path / "orm-read.db"
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "015")
    _seed_legacy_rows(db_path)
    command.upgrade(cfg, "head")

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession

    from src.db.repositories.lora_repo import LoraRepository
    from src.db.tables.runnable_mixin import RunnableStatus

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        loras = await LoraRepository(session).list_all()
        statuses = {lora.name: lora.status for lora in loras}
    await engine.dispose()

    assert statuses["completed-lora"] == RunnableStatus.COMPLETED
    assert statuses["running-job"] == RunnableStatus.ORPHAN
    assert statuses["failed-job"] == RunnableStatus.FAILED


def test_downgrade_from_head_recreates_legacy_table_shells(tmp_path) -> None:
    db_path = tmp_path / "downgrade.db"
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "015")

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    conn.close()
    assert {"jobs", "job_configs", "queue_entries", "trained_loras"} <= tables
