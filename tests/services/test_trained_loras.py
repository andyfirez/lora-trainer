"""Tests for trained LoRA catalog service."""

import pytest
from pathlib import Path

from src.db.tables.job import Job, JobStatus, JobType
from src.services.loras.discovery import LoraDiscoveryService
from src.services.loras.exceptions import TrainedLoraReproduceError
from src.services.loras.paths import resolve_trained_lora_paths, resolve_weights_path
from src.services.loras.service import TrainedLoraService
from src.trainer.config import TrainConfig


@pytest.mark.asyncio
async def test_create_trained_lora_from_completed_job(
    trained_lora_service: TrainedLoraService,
    session,
    minimal_training_yaml: str,
    storage_roots,
) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    output_rel = "trained"
    config = config.model_copy(update={"output_dir": output_rel, "lora_name": "demo_j1"})
    work_dir = storage_roots["lora"] / output_rel / "demo_j1"
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
    assert lora.relative_path == f"{output_rel}/demo_j1"
    assert resolve_weights_path(lora) == weights
    assert lora.name == "demo_j1"


@pytest.mark.asyncio
async def test_create_trained_lora_skips_missing_weights(
    trained_lora_service: TrainedLoraService,
    session,
    minimal_training_yaml: str,
    storage_roots,
) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    output_rel = "trained"
    config = config.model_copy(update={"output_dir": output_rel, "lora_name": "demo_j2"})
    work_dir = storage_roots["lora"] / output_rel / "demo_j2"
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


@pytest.mark.asyncio
async def test_list_loras_syncs_discovered_folder(
    trained_lora_service: TrainedLoraService,
    storage_roots,
) -> None:
    work_dir = storage_roots["lora"] / "discovered_lora"
    work_dir.mkdir()
    (work_dir / "discovered_lora.safetensors").write_bytes(b"weights")

    loras = await trained_lora_service.list_loras()
    assert any(lora.relative_path == "discovered_lora" for lora in loras)


@pytest.mark.asyncio
async def test_reproduce_requires_config_yaml(
    trained_lora_service: TrainedLoraService,
    storage_roots,
) -> None:
    work_dir = storage_roots["lora"] / "no_config"
    work_dir.mkdir()
    (work_dir / "no_config.safetensors").write_bytes(b"weights")
    discovered = LoraDiscoveryService().discover_lora_work_dirs()
    assert discovered

    loras = await trained_lora_service.list_loras()
    lora = next(item for item in loras if item.relative_path == "no_config")
    with pytest.raises(TrainedLoraReproduceError):
        await trained_lora_service.reproduce(lora.id)  # type: ignore[arg-type]


def test_resolve_trained_lora_paths(storage_roots, minimal_training_yaml: str) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    output_rel = "resolved"
    config = config.model_copy(update={"output_dir": output_rel, "lora_name": "demo"})
    work_dir = storage_roots["lora"] / output_rel / "demo"
    work_dir.mkdir(parents=True)
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
    assert paths.relative_path == f"{output_rel}/demo"


@pytest.mark.asyncio
async def test_sync_relocates_lora_when_moved_to_subfolder(
    trained_lora_service: TrainedLoraService,
    session,
    minimal_training_yaml: str,
    storage_roots,
) -> None:
    config = TrainConfig.from_yaml(minimal_training_yaml)
    output_rel = "trained"
    config = config.model_copy(update={"output_dir": output_rel, "lora_name": "demo_j1"})
    work_dir = storage_roots["lora"] / output_rel / "demo_j1"
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
    lora_id = lora.id

    moved_dir = storage_roots["lora"] / "anime" / "demo_j1"
    moved_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.rename(moved_dir)

    loras = await trained_lora_service.list_loras()
    assert len(loras) == 1
    assert loras[0].id == lora_id
    assert loras[0].name == "demo_j1"
    assert loras[0].relative_path == "anime/demo_j1"
    assert loras[0].job_id == job.id

    await session.refresh(job)
    assert Path(job.output_path).resolve() == moved_dir.resolve()


@pytest.mark.asyncio
async def test_sync_relocates_lora_when_folder_renamed(
    trained_lora_service: TrainedLoraService,
    storage_roots,
) -> None:
    work_dir = storage_roots["lora"] / "old_name"
    work_dir.mkdir()
    (work_dir / "old_name.safetensors").write_bytes(b"weights")

    loras = await trained_lora_service.list_loras()
    lora = next(item for item in loras if item.relative_path == "old_name")
    lora_id = lora.id
    lora_name = lora.name

    moved_dir = storage_roots["lora"] / "folder" / "new_name"
    moved_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.rename(moved_dir)

    loras_after = await trained_lora_service.list_loras()
    assert len(loras_after) == 1
    assert loras_after[0].id == lora_id
    assert loras_after[0].name == lora_name
    assert loras_after[0].relative_path == "folder/new_name"


@pytest.mark.asyncio
async def test_sync_creates_new_lora_when_no_stale_match(
    trained_lora_service: TrainedLoraService,
    storage_roots,
) -> None:
    work_dir = storage_roots["lora"] / "brand_new"
    work_dir.mkdir()
    (work_dir / "brand_new.safetensors").write_bytes(b"weights")

    loras = await trained_lora_service.list_loras()
    assert len(loras) == 1
    assert loras[0].relative_path == "brand_new"
    assert loras[0].name == "brand_new"


@pytest.mark.asyncio
async def test_sync_does_not_merge_ambiguous_stale_loras(
    trained_lora_service: TrainedLoraService,
    session,
    storage_roots,
) -> None:
    from src.db.tables.trained_lora import TrainedLora

    session.add_all(
        [
            TrainedLora(
                name="alpha-demo",
                relative_path="alpha/demo",
                weights_relpath="alpha/demo/demo.safetensors",
                base_model_name="unknown",
            ),
            TrainedLora(
                name="beta-demo",
                relative_path="beta/demo",
                weights_relpath="beta/demo/demo.safetensors",
                base_model_name="unknown",
            ),
        ]
    )
    await session.commit()

    work_dir = storage_roots["lora"] / "gamma" / "demo"
    work_dir.mkdir(parents=True)
    (work_dir / "demo.safetensors").write_bytes(b"weights")

    loras = await trained_lora_service.list_loras()
    assert len(loras) == 1
    assert loras[0].relative_path == "gamma/demo"
    assert loras[0].name == "demo"
