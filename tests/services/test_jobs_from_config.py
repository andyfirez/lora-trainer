import pytest
from src.db.tables.job import JobType
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService
from src.trainer.config import TrainConfig


@pytest.mark.asyncio
async def test_create_training_job_from_config(
    jobs_service: JobsService,
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    config = await config_service.create_config(
        name="training template",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )

    job = await jobs_service.create_from_config(config.id, name="my training run")

    assert job.job_type == JobType.TRAINING
    assert job.name == "my training run"
    assert job.config_id == config.id
    assert job.config_version == 1
    assert TrainConfig.from_yaml(job.config_yaml).lora_name == "lora"
    assert "_v1" not in job.config_yaml
    assert job.config_yaml == config.config_yaml
    assert job.output_path is None


@pytest.mark.asyncio
async def test_create_sampling_job_from_config(
    jobs_service: JobsService,
    config_service: JobConfigService,
    tmp_path,
) -> None:
    config = await config_service.create_config(
        name="sampling template",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"""
output_dir: {tmp_path.as_posix()}
sample_prompts:
  - test prompt
""",
    )

    job = await jobs_service.create_from_config(
        config.id,
        name="my sampling run",
    )

    assert job.job_type == JobType.SAMPLING
    assert job.name == "my sampling run"
    assert job.config_id == config.id
    assert jobs_service.get_lora_paths(job) == []
    assert job.output_path is not None
    assert f"job_{job.id}" in job.output_path


@pytest.mark.asyncio
async def test_create_job_from_config_uses_active_version_and_versioned_lora_name(
    jobs_service: JobsService,
    config_service: JobConfigService,
    minimal_training_yaml: str,
    training_dataset,
) -> None:
    config = await config_service.create_config(
        name="versioned training",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )
    updated_yaml = f"""base_model_name: changed
concepts:
  - dataset_id: {training_dataset.id}
"""
    config = await config_service.update_config(config.id, config_yaml=updated_yaml)

    job = await jobs_service.create_from_config(config.id, name="versioned run")

    assert config.active_version == 2
    assert job.config_version == 2
    assert TrainConfig.from_yaml(job.config_yaml).lora_name == "lora"
    assert "_v2" not in job.config_yaml

    from src.services.configs.versioning import apply_lora_version_to_train_config

    runtime = apply_lora_version_to_train_config(TrainConfig.from_yaml(job.config_yaml), job.config_version)
    assert runtime.lora_name == "lora_v2"
