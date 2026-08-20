"""Tests for SamplingService: create/enqueue/cancel lifecycle."""

from unittest.mock import MagicMock

import pytest
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableOperationNotSupportedError,
    RunnableValidationError,
)
from src.services.sampling.exceptions import SamplingPromptsNotConfiguredError
from src.services.sampling.service import SamplingService


@pytest.mark.asyncio
async def test_create_sampling_requires_absolute_output_dir(sampling_service: SamplingService) -> None:
    with pytest.raises(RunnableValidationError, match="absolute path"):
        await sampling_service.create_sampling(
            name="relative-output",
            config_yaml="output_dir: relative/path\n",
        )


@pytest.mark.asyncio
async def test_create_sampling_requires_prompts(sampling_service: SamplingService, sampling_output_dir) -> None:
    with pytest.raises(SamplingPromptsNotConfiguredError):
        await sampling_service.create_sampling(
            name="no-prompts",
            config_yaml=f"output_dir: {sampling_output_dir.as_posix()}\n",
        )


@pytest.mark.asyncio
async def test_create_sampling_sets_output_path(
    sampling_service: SamplingService,
    minimal_sampling_yaml: str,
) -> None:
    sampling = await sampling_service.create_sampling(name="my-sampling", config_yaml=minimal_sampling_yaml)
    assert sampling.status == RunnableStatus.DRAFT
    assert sampling.output_path is not None
    assert f"sampling_{sampling.id}" in sampling.output_path


@pytest.mark.asyncio
async def test_enqueue_sampling_sets_queued_status(
    sampling_service: SamplingService,
    minimal_sampling_yaml: str,
) -> None:
    sampling = await sampling_service.create_sampling(name="queue-me", config_yaml=minimal_sampling_yaml)
    queued = await sampling_service.enqueue_sampling(sampling.id)
    assert queued.status == RunnableStatus.QUEUED
    assert queued.queue_position == 1


@pytest.mark.asyncio
async def test_enqueue_already_queued_sampling_raises(
    sampling_service: SamplingService,
    minimal_sampling_yaml: str,
) -> None:
    sampling = await sampling_service.create_sampling(name="double-queue", config_yaml=minimal_sampling_yaml)
    await sampling_service.enqueue_sampling(sampling.id)
    with pytest.raises(RunnableAlreadyQueuedError):
        await sampling_service.enqueue_sampling(sampling.id)


@pytest.mark.asyncio
async def test_cancel_queued_sampling(sampling_service: SamplingService, minimal_sampling_yaml: str) -> None:
    sampling = await sampling_service.create_sampling(name="cancel-me", config_yaml=minimal_sampling_yaml)
    await sampling_service.enqueue_sampling(sampling.id)

    cancelled = await sampling_service.cancel_sampling(sampling.id)
    assert cancelled.status == RunnableStatus.CANCELLED
    assert cancelled.queue_position is None


def test_sample_file_path_rejects_missing_output_and_traversal(tmp_path) -> None:
    service = SamplingService(MagicMock())
    without_output = Sampling(id=1, name="s", config_yaml="x: 1\n")
    with pytest.raises(RunnableOperationNotSupportedError):
        service.sample_file_path(without_output, "a.png")

    sample = tmp_path / "a.png"
    sample.write_bytes(b"ok")
    sampling = Sampling(id=1, name="s", config_yaml="x: 1\n", output_path=str(tmp_path))
    assert service.sample_file_path(sampling, "a.png") == sample.resolve()
    with pytest.raises(RunnableOperationNotSupportedError):
        service.sample_file_path(sampling, "../secret.png")
