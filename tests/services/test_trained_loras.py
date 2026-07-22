import pytest
from pathlib import Path

from src.db.tables.job import Job, JobStatus, JobType
from src.services.loras.paths import resolve_trained_lora_paths
from src.services.loras.service import TrainedLoraService
from src.trainer.config import TrainConfig


@pytest.mark.asyncio
async def test_create_trained_lora_from_completed_job(
    trained_lora_service: TrainedLoraService,
    session,
    minimal_training_yaml: str,
    tmp_path: Path,
) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    config = config.model_copy(update={"output_dir": str(tmp_path), "lora_name": "demo_j1"})
    work_dir = tmp_path / "demo_j1"
    work_dir.mkdir(parents=True)
    weights = work_dir / "demo_j1.safetensors"
    weights.write_bytes(b"weights")

    job = Job(
        job_type=JobType.TRAINING,
        name="train",
        status=JobStatus.COMPLETED,
        config_yaml=config.to_yaml(),
        output_path=str(work_dir),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    lora = await trained_lora_service.create_from_completed_job(job)

    assert lora is not None
    assert lora.job_id == job.id
    assert lora.weights_path == str(weights)
    assert lora.work_dir == str(work_dir)
    assert lora.name == "demo_j1"


@pytest.mark.asyncio
async def test_create_trained_lora_skips_missing_weights(
    trained_lora_service: TrainedLoraService,
    session,
    minimal_training_yaml: str,
    tmp_path: Path,
) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    config = config.model_copy(update={"output_dir": str(tmp_path), "lora_name": "demo_j2"})
    work_dir = tmp_path / "demo_j2"
    work_dir.mkdir(parents=True)

    job = Job(
        job_type=JobType.TRAINING,
        name="train",
        status=JobStatus.COMPLETED,
        config_yaml=config.to_yaml(),
        output_path=str(work_dir),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    lora = await trained_lora_service.create_from_completed_job(job)
    assert lora is None


def test_resolve_trained_lora_paths(tmp_path: Path, minimal_training_yaml: str) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    config = config.model_copy(update={"output_dir": str(tmp_path), "lora_name": "demo"})
    work_dir = tmp_path / "demo"
    work_dir.mkdir()
    weights = work_dir / "demo.safetensors"
    weights.write_bytes(b"x")

    job = Job(
        id=1,
        job_type=JobType.TRAINING,
        name="train",
        config_yaml=config.to_yaml(),
        output_path=str(work_dir),
    )
    paths = resolve_trained_lora_paths(job)
    assert paths is not None
    assert paths.weights_path == weights
