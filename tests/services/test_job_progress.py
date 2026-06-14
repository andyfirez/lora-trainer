import pytest

from src.db.tables.job import JobStatus
from src.services.jobs.service import JobsService


@pytest.mark.asyncio
async def test_update_progress_initial_step_zero(
    jobs_service: JobsService,
    create_training_job,
    session,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)

    updated = await jobs_service._job_repo.update_progress(
        job,
        step=0,
        total=500,
        loss=0.0,
        avr_loss=0.0,
        epoch=0,
        epoch_total=10,
    )
    await session.commit()

    assert updated.progress_step == 0
    assert updated.progress_total == 500
    assert updated.progress_loss == 0.0
    assert updated.progress_avr_loss == 0.0
    assert updated.progress_epoch == 0
    assert updated.progress_epoch_total == 10


@pytest.mark.asyncio
async def test_update_progress_after_training_step(
    jobs_service: JobsService,
    create_training_job,
    session,
) -> None:
    job = await create_training_job()
    await jobs_service._job_repo.update_status(job, JobStatus.RUNNING, pid=1234)
    await jobs_service._job_repo.update_progress(
        job,
        step=0,
        total=500,
        loss=0.0,
        avr_loss=0.0,
        epoch=0,
        epoch_total=10,
    )

    updated = await jobs_service._job_repo.update_progress(
        job,
        step=42,
        total=500,
        loss=0.12,
        avr_loss=0.15,
        epoch=2,
        epoch_total=10,
    )
    await session.commit()

    assert updated.progress_step == 42
    assert updated.progress_total == 500
    assert updated.progress_loss == 0.12
    assert updated.progress_avr_loss == 0.15
    assert updated.progress_epoch == 2
