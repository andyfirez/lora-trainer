"""Tests for LoraService: create/enqueue/cancel/resume lifecycle."""

import pytest
from PIL import Image
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.loras.exceptions import LoraNameConflictError, LoraNotFoundError
from src.services.loras.service import LoraService
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableNotCancellableError,
    RunnableValidationError,
)


@pytest.mark.asyncio
async def test_create_lora_rejects_unprepared_dataset(
    lora_service: LoraService,
    datasets_service,
    storage_roots,
) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    Image.new("RGB", (1024, 1024)).save(image_dir / "test.png")
    dataset = await datasets_service.create_dataset(name="raw", relative_path="images")
    await datasets_service.update_dataset(
        dataset.id,
        name=None,
        relative_path=None,
        description=None,
        target_resolution=1024,
        update_target_resolution=True,
    )

    yaml_text = f"""base_model_name: test-model
resolution: 1024
concepts:
  - dataset_id: {dataset.id}
"""
    with pytest.raises(RunnableValidationError, match="not ready for training"):
        await lora_service.create_lora(name="bad-lora", config_yaml=yaml_text)


@pytest.mark.asyncio
async def test_create_lora_rejects_duplicate_name(lora_service: LoraService, minimal_training_yaml: str) -> None:
    await lora_service.create_lora(name="dup", config_yaml=minimal_training_yaml)
    with pytest.raises(LoraNameConflictError):
        await lora_service.create_lora(name="dup", config_yaml=minimal_training_yaml)


@pytest.mark.asyncio
async def test_get_lora_missing_raises(lora_service: LoraService) -> None:
    with pytest.raises(LoraNotFoundError):
        await lora_service.get_lora(999)


@pytest.mark.asyncio
async def test_enqueue_lora_sets_queued_status(lora_service: LoraService, minimal_training_yaml: str) -> None:
    lora = await lora_service.create_lora(name="queue-me", config_yaml=minimal_training_yaml)
    assert lora.status == RunnableStatus.DRAFT

    queued = await lora_service.enqueue_lora(lora.id)
    assert queued.status == RunnableStatus.QUEUED
    assert queued.queue_position == 1


@pytest.mark.asyncio
async def test_enqueue_already_queued_lora_raises(lora_service: LoraService, minimal_training_yaml: str) -> None:
    lora = await lora_service.create_lora(name="double-queue", config_yaml=minimal_training_yaml)
    await lora_service.enqueue_lora(lora.id)
    with pytest.raises(RunnableAlreadyQueuedError):
        await lora_service.enqueue_lora(lora.id)


@pytest.mark.asyncio
async def test_cancel_queued_lora(lora_service: LoraService, minimal_training_yaml: str) -> None:
    lora = await lora_service.create_lora(name="cancel-me", config_yaml=minimal_training_yaml)
    await lora_service.enqueue_lora(lora.id)

    cancelled = await lora_service.cancel_lora(lora.id)
    assert cancelled.status == RunnableStatus.CANCELLED
    assert cancelled.queue_position is None


@pytest.mark.asyncio
async def test_cancel_completed_lora_raises(
    lora_service: LoraService, minimal_training_yaml: str, storage_roots
) -> None:
    lora = await lora_service.create_lora(name="done", config_yaml=minimal_training_yaml)
    work_dir = storage_roots["lora"] / "done"
    work_dir.mkdir()
    (work_dir / "done.safetensors").write_bytes(b"fake-weights")
    lora.relative_path = "done"
    lora.weights_relpath = "done/done.safetensors"
    lora.status = RunnableStatus.COMPLETED
    with pytest.raises(RunnableNotCancellableError):
        await lora_service.cancel_lora(lora.id)


@pytest.mark.asyncio
async def test_reproduce_creates_new_lora_from_snapshot(lora_service: LoraService, minimal_training_yaml: str) -> None:
    source = await lora_service.create_lora(name="original", config_yaml=minimal_training_yaml)

    reproduced = await lora_service.reproduce(source.id, name="original-copy")
    assert reproduced.id != source.id
    assert reproduced.name == "original-copy"
    assert reproduced.status == RunnableStatus.DRAFT
    assert reproduced.config_yaml is not None
