"""Elapsed time helpers for jobs."""

from datetime import datetime, timezone

from src.db.tables.job import Job, JobStatus


def compute_elapsed_seconds(job: Job, *, now: datetime | None = None) -> float | None:
    current = now or datetime.now(timezone.utc)
    elapsed = job.accumulated_elapsed_seconds
    if job.status == JobStatus.RUNNING and job.running_started_at is not None:
        started_at = job.running_started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        elapsed += (current - started_at).total_seconds()
    if elapsed <= 0 and job.status in (JobStatus.PENDING, JobStatus.QUEUED):
        return None
    return elapsed
