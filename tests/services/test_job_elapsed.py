"""Tests for job elapsed time tracking."""

from datetime import datetime, timedelta, timezone

import pytest
from src.api.converters import to_job_response
from src.db.tables.job import JobStatus
from src.services.jobs.elapsed import compute_elapsed_seconds


@pytest.mark.asyncio
async def test_elapsed_accumulates_when_leaving_running(
    jobs_service,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    job.running_started_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()

    await jobs_service._job_repo.update_status(job, JobStatus.CANCELLED)

    stored = await jobs_service.get_job(job.id)
    assert stored.accumulated_elapsed_seconds == pytest.approx(120.0, rel=0.05)
    assert stored.running_started_at is None


@pytest.mark.asyncio
async def test_elapsed_preserves_accumulated_on_resume(
    jobs_service,
    create_training_job,
) -> None:
    job = await create_training_job()
    job.status = JobStatus.FAILED
    job.accumulated_elapsed_seconds = 300.0
    job.running_started_at = None
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()

    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=5678)

    stored = await jobs_service.get_job(job.id)
    assert stored.accumulated_elapsed_seconds == 300.0
    assert stored.running_started_at is not None


@pytest.mark.asyncio
async def test_elapsed_resets_on_fresh_enqueue(
    jobs_service,
    create_training_job,
) -> None:
    job = await create_training_job()
    job.accumulated_elapsed_seconds = 500.0
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()
    await jobs_service._job_repo.update_status(job, JobStatus.CANCELLED)
    await jobs_service.enqueue_job(job.id)

    stored = await jobs_service.get_job(job.id)
    assert stored.accumulated_elapsed_seconds == 0.0
    assert stored.running_started_at is None


@pytest.mark.asyncio
async def test_elapsed_seconds_in_job_response(
    jobs_service,
    create_training_job,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    job.accumulated_elapsed_seconds = 60.0
    job.running_started_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()
    await jobs_service._job_repo._session.refresh(job)

    response = to_job_response(job, jobs_service)
    assert response.elapsed_seconds is not None
    assert response.elapsed_seconds >= 85.0


@pytest.mark.asyncio
async def test_compute_elapsed_seconds_completed_job(
    jobs_service,
    create_training_job,
) -> None:
    job = await create_training_job()
    job.accumulated_elapsed_seconds = 42.5
    job.running_started_at = None
    await jobs_service._job_repo.update_status(job, JobStatus.COMPLETED)

    assert compute_elapsed_seconds(job) == 42.5
