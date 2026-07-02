"""Tests for job API response conversion."""

import pytest

from src.api.converters import to_job_response
from src.db.tables.job import JobType
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService


@pytest.mark.asyncio
async def test_training_job_response_includes_sampling_config_id(
    jobs_service: JobsService,
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    sampling_config = await config_service.create_config(
        name="sampling template",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - test prompt\n",
    )
    training_config = await config_service.create_config(
        name="training template",
        config_type=ConfigType.TRAINING,
        config_yaml=f"{minimal_training_yaml}sampling_config_id: {sampling_config.id}\n",
    )
    job = await jobs_service.create_from_config(training_config.id, name="my training run")

    response = to_job_response(job, jobs_service)

    assert response.job_type == JobType.TRAINING
    assert response.training is not None
    assert response.training.sampling_config_id == sampling_config.id


@pytest.mark.asyncio
async def test_training_job_response_sampling_config_id_null_when_unset(
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

    assert response.training is not None
    assert response.training.sampling_config_id is None
