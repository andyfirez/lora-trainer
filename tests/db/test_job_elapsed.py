"""Tests for job elapsed time tracking."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.api.converters import to_job_response
from src.db.elapsed import compute_elapsed_seconds
from src.db.tables.job import JobStatus
from src.services.jobs.service import JobsService


@pytest.mark.asyncio
async def test_update_status_accumulates_elapsed_when_leaving_running(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=120)

    with patch("src.db.repositories.job_repo.datetime") as mock_datetime:
        mock_datetime.now.return_value = t0
        await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)

        mock_datetime.now.return_value = t1
        await jobs_service._job_repo.update_status(job, JobStatus.CANCELLED)

    assert job.running_started_at is None
    assert job.accumulated_elapsed_seconds == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_elapsed_running_includes_current_session(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    session_start = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    job.accumulated_elapsed_seconds = 30.0
    job.running_started_at = session_start
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()

    now = session_start + timedelta(seconds=45)
    elapsed = compute_elapsed_seconds(job, now=now)

    assert elapsed == pytest.approx(75.0)


@pytest.mark.asyncio
async def test_elapsed_preserved_on_resume(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    job.accumulated_elapsed_seconds = 200.0
    job.running_started_at = None
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()

    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=5678)

    assert job.accumulated_elapsed_seconds == pytest.approx(200.0)
    assert job.running_started_at is not None


@pytest.mark.asyncio
async def test_elapsed_seconds_in_job_response(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()
    started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    job.accumulated_elapsed_seconds = 10.0
    job.running_started_at = started_at
    jobs_service._job_repo._session.add(job)
    await jobs_service._job_repo._session.flush()

    with patch("src.db.elapsed.datetime") as mock_datetime:
        now = started_at + timedelta(seconds=50)
        mock_datetime.now.return_value = now
        response = to_job_response(job, jobs_service)

    assert response.elapsed_seconds == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_elapsed_seconds_null_before_first_run(
    jobs_service: JobsService,
    create_training_job,
) -> None:
    job = await create_training_job()

    response = to_job_response(job, jobs_service)

    assert response.elapsed_seconds is None
    assert compute_elapsed_seconds(job) is None
