"""SQLite schema migrations for existing databases."""

from sqlalchemy.ext.asyncio import AsyncConnection

_TRAINING_JOB_COLUMNS: list[tuple[str, str]] = [
    ("progress_loss", "REAL"),
    ("progress_avr_loss", "REAL"),
    ("progress_epoch", "INTEGER"),
    ("progress_epoch_total", "INTEGER"),
    ("cache_progress_step", "INTEGER"),
    ("cache_progress_total", "INTEGER"),
    ("sampling_status", "TEXT"),
    ("sampling_step", "INTEGER"),
    ("sampling_total", "INTEGER"),
    ("last_checkpoint_path", "TEXT"),
    ("last_checkpoint_epoch", "INTEGER"),
    ("last_checkpoint_step", "INTEGER"),
    ("resume_checkpoint_path", "TEXT"),
    ("resume_from_epoch", "INTEGER"),
    ("resume_from_step", "INTEGER"),
    ("save_checkpoint_requested", "INTEGER DEFAULT 0"),
]

_QUEUE_ENTRY_COLUMNS: list[tuple[str, str]] = [
    ("item_type", "TEXT DEFAULT 'training'"),
    ("item_id", "INTEGER"),
]

_SAMPLING_RUN_COLUMNS: list[tuple[str, str]] = [
    ("name", "TEXT NOT NULL DEFAULT 'Sampling Run'"),
    ("config_yaml", "TEXT NOT NULL DEFAULT ''"),
    ("lora_paths_yaml", "TEXT NOT NULL DEFAULT '[]'"),
    ("status", "TEXT NOT NULL DEFAULT 'pending'"),
    ("source_job_id", "INTEGER"),
    ("output_path", "TEXT"),
    ("log_path", "TEXT"),
    ("pid", "INTEGER"),
    ("error_message", "TEXT"),
    ("progress_status", "TEXT"),
    ("progress_step", "INTEGER"),
    ("progress_total", "INTEGER"),
    ("created_at", "TEXT"),
    ("updated_at", "TEXT"),
]


def _apply_training_job_migrations(sync_conn) -> None:
    if not _table_exists(sync_conn, "training_jobs"):
        return
    rows = sync_conn.exec_driver_sql("PRAGMA table_info(training_jobs)").fetchall()
    existing = {row[1] for row in rows}
    for column_name, column_type in _TRAINING_JOB_COLUMNS:
        if column_name not in existing:
            sync_conn.exec_driver_sql(
                f"ALTER TABLE training_jobs ADD COLUMN {column_name} {column_type}"
            )


def _table_exists(sync_conn, table_name: str) -> bool:
    row = sync_conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _queue_entry_columns(sync_conn) -> set[str]:
    rows = sync_conn.exec_driver_sql("PRAGMA table_info(queue_entries)").fetchall()
    return {row[1] for row in rows}


def _rebuild_queue_entries_table(sync_conn) -> None:
    """Drop legacy job_id column by recreating queue_entries with item_type/item_id."""
    sync_conn.exec_driver_sql(
        """
        CREATE TABLE queue_entries_new (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL DEFAULT 'training',
            item_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            added_at DATETIME NOT NULL
        )
        """
    )
    sync_conn.exec_driver_sql(
        """
        INSERT INTO queue_entries_new (id, item_type, item_id, position, added_at)
        SELECT
            id,
            CASE
                WHEN LOWER(COALESCE(item_type, 'training')) = 'sampling' THEN 'sampling'
                ELSE 'training'
            END,
            COALESCE(item_id, job_id),
            position,
            added_at
        FROM queue_entries
        """
    )
    sync_conn.exec_driver_sql("DROP TABLE queue_entries")
    sync_conn.exec_driver_sql("ALTER TABLE queue_entries_new RENAME TO queue_entries")
    sync_conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_queue_entries_item_type ON queue_entries (item_type)"
    )
    sync_conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_queue_entries_item_id ON queue_entries (item_id)"
    )
    sync_conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_queue_entries_position ON queue_entries (position)"
    )


def _apply_queue_entry_migrations(sync_conn) -> None:
    if not _table_exists(sync_conn, "queue_entries"):
        return
    existing = _queue_entry_columns(sync_conn)
    for column_name, column_type in _QUEUE_ENTRY_COLUMNS:
        if column_name not in existing:
            sync_conn.exec_driver_sql(
                f"ALTER TABLE queue_entries ADD COLUMN {column_name} {column_type}"
            )
    existing = _queue_entry_columns(sync_conn)
    if "job_id" in existing:
        if "item_id" in existing:
            sync_conn.exec_driver_sql(
                "UPDATE queue_entries SET item_id = job_id WHERE item_id IS NULL"
            )
        if "item_type" in existing:
            sync_conn.exec_driver_sql(
                "UPDATE queue_entries SET item_type = 'training' WHERE item_type IS NULL"
            )
        _rebuild_queue_entries_table(sync_conn)


def _apply_sampling_run_migrations(sync_conn) -> None:
    if not _table_exists(sync_conn, "sampling_runs"):
        return
    rows = sync_conn.exec_driver_sql("PRAGMA table_info(sampling_runs)").fetchall()
    existing = {row[1] for row in rows}
    for column_name, column_type in _SAMPLING_RUN_COLUMNS:
        if column_name not in existing:
            sync_conn.exec_driver_sql(
                f"ALTER TABLE sampling_runs ADD COLUMN {column_name} {column_type}"
            )


async def migrate_schema(conn: AsyncConnection) -> None:
    await conn.run_sync(_apply_training_job_migrations)
    await conn.run_sync(_apply_queue_entry_migrations)
    await conn.run_sync(_apply_sampling_run_migrations)
