from pathlib import Path

import pytest
from src.db.tables.job import JobStatus, JobType
from src.db.tables.job_config import ConfigType
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.services.configs.exceptions import JobConfigValidationError
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService
from src.services.sampling.exceptions import (
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)


def test_resolve_sampling_output_path_absolute(tmp_path: Path) -> None:
    output_dir = tmp_path / "samples"
    config = SamplingConfig(output_dir=str(output_dir))
    path = resolve_sampling_output_path(config, job_id=42)
    assert path == output_dir.resolve() / "job_42"


def test_resolve_sampling_output_path_requires_absolute() -> None:
    config = SamplingConfig(output_dir="relative/path")
    with pytest.raises(ValueError, match="absolute path"):
        resolve_sampling_output_path(config, job_id=1)


def test_resolve_sampling_output_path_requires_non_empty() -> None:
    config = SamplingConfig(output_dir="")
    with pytest.raises(ValueError, match="output_dir is required"):
        resolve_sampling_output_path(config, job_id=1)


@pytest.mark.asyncio
async def test_create_from_config_resolves_lora_paths_from_request(
    jobs_service: JobsService,
    config_service: JobConfigService,
    minimal_sampling_yaml: str,
    sampling_output_dir: Path,
    tmp_path: Path,
) -> None:
    lora_path = tmp_path / "custom.safetensors"
    lora_path.write_bytes(b"lora")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=minimal_sampling_yaml,
    )

    sampling_job = await jobs_service.create_from_config(
        sampling_config.id,
        lora_paths=[str(lora_path)],
    )

    assert sampling_job.job_type == JobType.SAMPLING
    assert jobs_service.get_lora_paths(sampling_job) == [str(lora_path)]
    assert sampling_job.output_path == str(sampling_output_dir / f"job_{sampling_job.id}")


@pytest.mark.asyncio
async def test_create_from_config_standalone_resolves_lora_paths_from_config(
    jobs_service: JobsService,
    config_service: JobConfigService,
    sampling_output_dir: Path,
    tmp_path: Path,
) -> None:
    lora_a = tmp_path / "a.safetensors"
    lora_b = tmp_path / "b.safetensors"
    lora_a.write_bytes(b"a")
    lora_b.write_bytes(b"b")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=f"""
output_dir: {sampling_output_dir.as_posix()}
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
    minimal_sampling_yaml: str,
    sampling_output_dir: Path,
) -> None:
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=minimal_sampling_yaml,
    )

    sampling_job = await jobs_service.create_from_config(sampling_config.id)

    assert jobs_service.get_lora_paths(sampling_job) == []
    assert sampling_job.output_path == str(sampling_output_dir / f"job_{sampling_job.id}")


@pytest.mark.asyncio
async def test_create_from_config_rejects_missing_sample_prompts(
    jobs_service: JobsService,
    config_service: JobConfigService,
    sampling_output_dir: Path,
    tmp_path: Path,
) -> None:
    lora_path = tmp_path / "model.safetensors"
    lora_path.write_bytes(b"lora")
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=f'output_dir: {sampling_output_dir.as_posix()}\n',
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
    sampling_output_dir: Path,
) -> None:
    log_path = tmp_path / "sampling_job_1.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    from src.db.tables.job import Job

    sampling_job = Job(
        job_type=JobType.SAMPLING,
        name="sample",
        config_yaml=f"""output_dir: {sampling_output_dir.as_posix()}
parameters:
  prompt:
    mode: fixed
    value: test
""",
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
    minimal_sampling_yaml: str,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.safetensors"
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml=minimal_sampling_yaml,
    )

    with pytest.raises(SamplingLoRAPathNotFoundError):
        await jobs_service.create_from_config(
            sampling_config.id,
            lora_paths=[str(missing_path)],
        )


@pytest.mark.asyncio
async def test_create_sampling_config_requires_absolute_output_dir(
    config_service: JobConfigService,
) -> None:
    with pytest.raises(JobConfigValidationError, match="absolute path"):
        await config_service.create_config(
            name="sampling",
            config_type=ConfigType.SAMPLING,
            config_yaml="""output_dir: /tmp/out
parameters:
  prompt:
    mode: fixed
    value: test
""",
        )


@pytest.mark.asyncio
async def test_create_sampling_config_requires_output_dir(
    config_service: JobConfigService,
) -> None:
    with pytest.raises(JobConfigValidationError, match="output_dir is required"):
        await config_service.create_config(
            name="sampling",
            config_type=ConfigType.SAMPLING,
            config_yaml='output_dir: ""\nparameters:\n  prompt:\n    mode: fixed\n    value: test\n',
        )
