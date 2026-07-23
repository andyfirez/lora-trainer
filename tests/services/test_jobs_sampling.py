from pathlib import Path

import pytest
from src.db.tables.job import JobStatus, JobType
from src.db.tables.job_config import ConfigType
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService
from src.services.sampling.exceptions import (
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)
from src.storage.paths import StoragePaths
from src.trainer.config import TrainConfig


def test_resolve_sampling_output_path_standalone(storage_roots) -> None:
    config = SamplingConfig(output_dir="")
    path = resolve_sampling_output_path(config, job_id=42, source_train_config=None)
    assert path == storage_roots["lora"] / "samples" / "job_42"


def test_resolve_sampling_output_path_train_linked(storage_roots) -> None:
    config = SamplingConfig(output_dir="ignored")
    train_config = TrainConfig(output_dir="output", lora_name="demo")
    path = resolve_sampling_output_path(config, job_id=42, source_train_config=train_config)
    assert path == storage_roots["lora"] / "output" / "demo" / "samples"


@pytest.mark.asyncio
async def test_create_from_config_train_linked_resolves_lora_paths(
    jobs_service: JobsService,
    config_service: JobConfigService,
    create_training_job,
    storage_roots,
    tmp_path,
) -> None:
    lora_path = tmp_path / "custom.safetensors"
    lora_path.write_bytes(b"lora")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - test prompt\n",
    )
    training_job = await create_training_job()

    sampling_job = await jobs_service.create_from_config(
        sampling_config.id,
        lora_paths=[str(lora_path)],
        source_job_id=training_job.id,
    )

    train_config = TrainConfig.from_yaml(training_job.config_yaml)
    work_dir = StoragePaths.resolve_training_work_dir(train_config.output_dir, train_config.lora_name)
    assert sampling_job.job_type == JobType.SAMPLING
    assert sampling_job.source_job_id == training_job.id
    assert jobs_service.get_lora_paths(sampling_job) == [str(lora_path)]
    assert sampling_job.output_path == str(work_dir / "samples")


@pytest.mark.asyncio
async def test_create_from_config_standalone_resolves_lora_paths_from_config(
    jobs_service: JobsService,
    config_service: JobConfigService,
    storage_roots,
    tmp_path,
) -> None:
    lora_a = tmp_path / "a.safetensors"
    lora_b = tmp_path / "b.safetensors"
    lora_a.write_bytes(b"a")
    lora_b.write_bytes(b"b")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"""
output_dir: ""
lora_paths:
  - {lora_a.as_posix()}
  - {lora_b.as_posix()}
parameters:
  lora_path:
    mode: vary
    values:
      - {lora_a.as_posix()}
      - {lora_b.as_posix()}
  prompt:
    mode: fixed
    value: test prompt
""",
    )

    sampling_job = await jobs_service.create_from_config(sampling_config.id)

    assert [Path(p) for p in jobs_service.get_lora_paths(sampling_job)] == [lora_a, lora_b]


@pytest.mark.asyncio
async def test_create_from_config_standalone_has_empty_lora_paths(
    jobs_service: JobsService,
    config_service: JobConfigService,
    storage_roots,
) -> None:
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml='output_dir: ""\nsample_prompts:\n  - test prompt\n',
    )

    sampling_job = await jobs_service.create_from_config(
        sampling_config.id,
    )

    assert jobs_service.get_lora_paths(sampling_job) == []
    assert sampling_job.output_path == str(storage_roots["lora"] / "samples" / f"job_{sampling_job.id}")


@pytest.mark.asyncio
async def test_create_from_config_train_linked_auto_resolves_checkpoints(
    jobs_service: JobsService,
    config_service: JobConfigService,
    training_dataset,
    storage_roots,
) -> None:
    output_rel = "output"
    training_config = await config_service.create_config(
        name="training",
        config_type=ConfigType.TRAINING,
        config_yaml=f"""
base_model_name: test-model
output_dir: {output_rel}
lora_name: demo
output_format: safetensors
concepts:
  - dataset_id: {training_dataset.id}
""",
    )
    training_job = await jobs_service.create_from_config(training_config.id)
    train_config = TrainConfig.from_yaml(training_job.config_yaml)
    work_dir = storage_roots["lora"] / output_rel / train_config.lora_name
    work_dir.mkdir(parents=True)
    epoch_path = work_dir / f"{train_config.lora_name}_epoch1.safetensors"
    step_path = work_dir / f"{train_config.lora_name}_step10.safetensors"
    epoch_path.write_bytes(b"epoch")
    step_path.write_bytes(b"step")

    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - test prompt\n",
    )

    sampling_job = await jobs_service.create_from_config(
        sampling_config.id,
        source_job_id=training_job.id,
    )

    assert jobs_service.get_lora_paths(sampling_job) == [str(epoch_path), str(step_path)]
    assert sampling_job.output_path == str(work_dir / "samples")


@pytest.mark.asyncio
async def test_create_from_config_rejects_missing_sample_prompts(
    jobs_service: JobsService,
    config_service: JobConfigService,
    storage_roots,
    tmp_path,
) -> None:
    lora_path = tmp_path / "model.safetensors"
    lora_path.write_bytes(b"lora")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml='output_dir: ""\n',
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
    training_dataset,
    session,
    storage_roots,
) -> None:
    output_rel = "output"

    sampling_config = await config_service.create_config(
        name="post-train sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="""
sample_prompts:
  - test prompt
""",
    )
    config_yaml = f"""
base_model_name: test-model
output_dir: {output_rel}
lora_name: demo
output_format: safetensors
checkpointing_enabled: true
sampling_enabled: true
sampling_config_id: {sampling_config.id}
concepts:
  - dataset_id: {training_dataset.id}
"""
    training_config = await config_service.create_config(
        name="training",
        config_type=ConfigType.TRAINING,
        config_yaml=config_yaml,
    )
    training_job = await jobs_service.create_from_config(training_config.id)
    train_config = TrainConfig.from_yaml(training_job.config_yaml)
    work_dir = storage_roots["lora"] / output_rel / train_config.lora_name
    work_dir.mkdir(parents=True)
    epoch_path = work_dir / f"{train_config.lora_name}_epoch1.safetensors"
    step_path = work_dir / f"{train_config.lora_name}_step10.safetensors"
    final_path = work_dir / f"{train_config.lora_name}.safetensors"
    epoch_path.write_bytes(b"epoch")
    step_path.write_bytes(b"step")
    final_path.write_bytes(b"final")
    await jobs_service._job_repo.update_status(training_job, JobStatus.COMPLETED)
    await session.commit()

    sampling_job = await jobs_service.create_auto_sampling_for_training_job(training_job)

    assert sampling_job is not None
    assert sampling_job.status == JobStatus.QUEUED
    assert jobs_service.get_lora_paths(sampling_job) == [str(epoch_path), str(step_path)]
    assert sampling_job.output_path == str(work_dir / "samples")
    queue_entry = await jobs_service._queue_repo.get_by_job_id(sampling_job.id)
    assert queue_entry is not None
    assert queue_entry.job_id == sampling_job.id
