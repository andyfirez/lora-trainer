"""Helpers for computing job elapsed time."""

from datetime import datetime, timezone

from src.db.tables.job import Job, JobStatus


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def compute_elapsed_seconds(job: Job, *, now: datetime | None = None) -> float | None:
    """Return elapsed running time in seconds, or None if the job never ran."""
    accumulated = job.accumulated_elapsed_seconds or 0.0
    has_running_session = job.running_started_at is not None
    has_elapsed = accumulated > 0 or has_running_session

    if job.status == JobStatus.RUNNING and has_running_session:
        current = now or datetime.now(timezone.utc)
        started_at = _ensure_utc(job.running_started_at)  # type: ignore[arg-type]
        return accumulated + (current - started_at).total_seconds()

    if has_elapsed:
        return accumulated

    return None
