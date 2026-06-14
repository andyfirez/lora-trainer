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


def _apply_training_job_migrations(sync_conn) -> None:
    rows = sync_conn.exec_driver_sql("PRAGMA table_info(training_jobs)").fetchall()
    existing = {row[1] for row in rows}
    for column_name, column_type in _TRAINING_JOB_COLUMNS:
        if column_name not in existing:
            sync_conn.exec_driver_sql(
                f"ALTER TABLE training_jobs ADD COLUMN {column_name} {column_type}"
            )


async def migrate_schema(conn: AsyncConnection) -> None:
    await conn.run_sync(_apply_training_job_migrations)
