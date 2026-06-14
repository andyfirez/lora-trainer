import pytest

from src.db.tables.job import JobStatus, JobType
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService
from src.services.sampling.exceptions import SamplingLoRAPathNotFoundError, SamplingPromptsNotConfiguredError


@pytest.mark.asyncio
async def test_create_from_config_accepts_arbitrary_lora_path(
    jobs_service: JobsService,
    config_service: JobConfigService,
    create_training_job,
    tmp_path,
) -> None:
    lora_path = tmp_path / "custom.safetensors"
    lora_path.write_bytes(b"lora")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"output_dir: {tmp_path.as_posix()}\nlora_name: demo\nsample_prompts:\n  - test prompt\n",
    )
    training_job = await create_training_job()

    sampling_job = await jobs_service.create_from_config(
        sampling_config.id,
        lora_paths=[str(lora_path)],
        source_job_id=training_job.id,
    )

    assert sampling_job.job_type == JobType.SAMPLING
    assert sampling_job.source_job_id == training_job.id
    assert jobs_service.get_lora_paths(sampling_job) == [str(lora_path)]
    assert sampling_job.output_path is not None


@pytest.mark.asyncio
async def test_create_from_config_rejects_missing_sample_prompts(
    jobs_service: JobsService,
    config_service: JobConfigService,
    tmp_path,
) -> None:
    lora_path = tmp_path / "model.safetensors"
    lora_path.write_bytes(b"lora")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"output_dir: {tmp_path.as_posix()}\nlora_name: demo\n",
    )

    with pytest.raises(SamplingPromptsNotConfiguredError):
        await jobs_service.create_from_config(
            sampling_config.id,
            lora_paths=[str(lora_path)],
        )


@pytest.mark.asyncio
async def test_get_job_logs_tail(
    jobs_service: JobsService,
    session,
    tmp_path,
) -> None:
    log_path = tmp_path / "sampling_job_1.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    from src.db.tables.job import Job

    sampling_job = Job(
        job_type=JobType.SAMPLING,
        name="sample",
        config_yaml="base_model_name: x",
        lora_paths_yaml="[]",
        log_path=str(log_path),
    )
    session.add(sampling_job)
    await session.commit()
    await session.refresh(sampling_job)

    lines = await jobs_service.get_job_logs(sampling_job.id, tail=2)

    assert lines == ["line2", "line3"]


@pytest.mark.asyncio
async def test_create_from_config_rejects_missing_lora_path(
    jobs_service: JobsService,
    config_service: JobConfigService,
    tmp_path,
) -> None:
    missing_path = tmp_path / "missing.safetensors"
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="base_model_name: x\nsample_prompts:\n  - test\n",
    )

    with pytest.raises(SamplingLoRAPathNotFoundError):
        await jobs_service.create_from_config(
            sampling_config.id,
            lora_paths=[str(missing_path)],
        )


@pytest.mark.asyncio
async def test_auto_sampling_enqueues_intermediate_checkpoints(
    jobs_service: JobsService,
    config_service: JobConfigService,
    session,
    tmp_path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir = output_dir / "demo"
    work_dir.mkdir(parents=True)
    epoch_path = work_dir / "demo_epoch1.safetensors"
    step_path = work_dir / "demo_step10.safetensors"
    final_path = work_dir / "demo.safetensors"
    epoch_path.write_bytes(b"epoch")
    step_path.write_bytes(b"step")
    final_path.write_bytes(b"final")

    sampling_config = await config_service.create_config(
        name="post-train sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"""
output_dir: {output_dir.as_posix()}
lora_name: demo
sample_prompts:
  - test prompt
""",
    )
    config_yaml = f"""
output_dir: {output_dir.as_posix()}
lora_name: demo
output_format: safetensors
sample_after_training: true
post_training_sampling_config_id: {sampling_config.id}
"""
    training_config = await config_service.create_config(
        name="training",
        config_type=ConfigType.TRAINING,
        config_yaml=config_yaml,
    )
    training_job = await jobs_service.create_from_config(training_config.id)
    await jobs_service._job_repo.update_status(training_job, JobStatus.COMPLETED)
    await session.commit()

    sampling_job = await jobs_service.create_auto_sampling_for_training_job(training_job)

    assert sampling_job is not None
    assert sampling_job.status == JobStatus.QUEUED
    assert jobs_service.get_lora_paths(sampling_job) == [str(epoch_path), str(step_path)]
    queue_entry = await jobs_service._queue_repo.get_by_job_id(sampling_job.id)
    assert queue_entry is not None
    assert queue_entry.job_id == sampling_job.id
