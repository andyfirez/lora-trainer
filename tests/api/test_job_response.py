"""Tests for job API response conversion."""

import pytest
from src.api.converters import to_job_response
from src.db.tables.job import JobType
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService


@pytest.mark.asyncio
async def test_training_job_response_has_training_details(
    jobs_service: JobsService,
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    training_config = await config_service.create_config(
        name="training template",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )
    job = await jobs_service.create_from_config(training_config.id, name="my training run")

    response = to_job_response(job, jobs_service)

    assert response.job_type == JobType.TRAINING
    assert response.training is not None
    assert response.training.save_checkpoint_requested is False
