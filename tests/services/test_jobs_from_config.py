import pytest

from src.db.tables.job import JobType
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService


@pytest.mark.asyncio
async def test_create_training_job_from_config(
    jobs_service: JobsService,
    config_service: JobConfigService,
) -> None:
    config = await config_service.create_config(
        name="training template",
        config_type=ConfigType.TRAINING,
        config_yaml="base_model_name: stabilityai/stable-diffusion-xl-base-1.0",
    )

    job = await jobs_service.create_from_config(config.id, name="my training run")

    assert job.job_type == JobType.TRAINING
    assert job.name == "my training run"
    assert job.config_id == config.id
    assert job.config_yaml == config.config_yaml
    assert job.output_path is None


@pytest.mark.asyncio
async def test_create_sampling_job_from_config(
    jobs_service: JobsService,
    config_service: JobConfigService,
    tmp_path,
) -> None:
    lora_path = tmp_path / "model.safetensors"
    lora_path.write_bytes(b"lora")
    config = await config_service.create_config(
        name="sampling template",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"""
output_dir: {tmp_path.as_posix()}
lora_name: demo
sample_prompts:
  - test prompt
""",
    )

    job = await jobs_service.create_from_config(
        config.id,
        name="my sampling run",
        lora_paths=[str(lora_path)],
    )

    assert job.job_type == JobType.SAMPLING
    assert job.name == "my sampling run"
    assert job.config_id == config.id
    assert jobs_service.get_lora_paths(job) == [str(lora_path)]
    assert job.output_path is not None
    assert f"job_{job.id}" in job.output_path
