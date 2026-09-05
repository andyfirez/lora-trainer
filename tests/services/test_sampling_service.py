"""Tests for SamplingService: create/enqueue/cancel lifecycle."""

from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from src.db.tables.runnable_mixin import RunnableStatus


@contextmanager
def turing_gpu():
    with (
        patch("src.gpu.validation.torch.cuda.is_available", return_value=True),
        patch("src.gpu.validation.torch.cuda.get_device_capability", return_value=(7, 5)),
        patch("src.gpu.validation.torch.cuda.get_device_name", return_value="RTX 2070"),
        patch("src.gpu.validation.torch.cuda.is_bf16_supported", return_value=False),
        patch("src.gpu.validation.is_xformers_available", return_value=True),
    ):
        yield
from src.db.tables.sampling import Sampling, SamplingKind
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableOperationNotSupportedError,
    RunnableValidationError,
)
from src.services.sampling.exceptions import LivePreviewNotReadyError, SamplingPromptsNotConfiguredError
from src.services.sampling.service import SamplingService


@pytest.mark.asyncio
async def test_create_sampling_rejects_path_outside_project_root(
    sampling_service: SamplingService,
) -> None:
    with pytest.raises(RunnableValidationError, match="escapes project directory"):
        await sampling_service.create_sampling(
            name="bad-output",
            config={"output_dir": "../../outside"},
        )


@pytest.mark.asyncio
async def test_create_sampling_defaults_empty_output_dir(
    sampling_service: SamplingService,
    minimal_sampling_config_no_output: dict,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sampling = await sampling_service.create_sampling(
        name="default-output",
        config=minimal_sampling_config_no_output,
    )
    assert sampling.output_path is not None
    assert sampling.output_path.startswith(str((tmp_path / "output").resolve()))
    assert sampling.output_path.endswith(date.today().isoformat())


@pytest.mark.asyncio
async def test_create_sampling_requires_prompts(sampling_service: SamplingService, sampling_output_dir) -> None:
    with pytest.raises(SamplingPromptsNotConfiguredError):
        await sampling_service.create_sampling(
            name="no-prompts",
            config={"output_dir": sampling_output_dir.as_posix()},
        )


@pytest.mark.asyncio
async def test_create_sampling_sets_output_path(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    sampling = await sampling_service.create_sampling(name="my-sampling", config=minimal_sampling_config)
    assert sampling.status == RunnableStatus.DRAFT
    assert sampling.output_path is not None
    assert sampling.output_path.endswith(date.today().isoformat())


@pytest.mark.asyncio
async def test_enqueue_sampling_sets_queued_status(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    sampling = await sampling_service.create_sampling(name="queue-me", config=minimal_sampling_config)
    queued = await sampling_service.enqueue_sampling(sampling.id)
    assert queued.status == RunnableStatus.QUEUED
    assert queued.queue_position == 1


@pytest.mark.asyncio
async def test_enqueue_already_queued_sampling_raises(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    sampling = await sampling_service.create_sampling(name="double-queue", config=minimal_sampling_config)
    await sampling_service.enqueue_sampling(sampling.id)
    with pytest.raises(RunnableAlreadyQueuedError):
        await sampling_service.enqueue_sampling(sampling.id)


@pytest.mark.asyncio
async def test_cancel_queued_sampling(sampling_service: SamplingService, minimal_sampling_config: dict) -> None:
    sampling = await sampling_service.create_sampling(name="cancel-me", config=minimal_sampling_config)
    await sampling_service.enqueue_sampling(sampling.id)

    cancelled = await sampling_service.cancel_sampling(sampling.id)
    assert cancelled.status == RunnableStatus.CANCELLED
    assert cancelled.queue_position is None


def test_sample_file_path_rejects_missing_output_and_traversal(tmp_path) -> None:
    service = SamplingService(MagicMock())
    without_output = Sampling(id=1, name="s", config={"x": 1})
    with pytest.raises(RunnableOperationNotSupportedError):
        service.sample_file_path(without_output, "a.png")

    sampling = Sampling(id=1, name="s", config={"x": 1}, output_path=str(tmp_path))
    with pytest.raises(RunnableOperationNotSupportedError):
        service.sample_file_path(sampling, "../secret.png")


def test_live_preview_path_requires_file(tmp_path) -> None:
    service = SamplingService(MagicMock())
    without_output = Sampling(id=2, name="s", config={"x": 1})
    with pytest.raises(LivePreviewNotReadyError):
        service.live_preview_path(without_output)

    sampling = Sampling(id=2, name="s", config={"x": 1}, output_path=str(tmp_path))
    with pytest.raises(LivePreviewNotReadyError):
        service.live_preview_path(sampling)

    preview = tmp_path / "live_preview.jpg"
    preview.write_bytes(b"jpeg")
    assert service.live_preview_path(sampling) == preview.resolve()


@pytest.mark.asyncio
async def test_create_sampling_sets_kind(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    from src.db.tables.sampling import SamplingKind

    sampling = await sampling_service.create_sampling(
        name="exp",
        config=minimal_sampling_config,
        kind=SamplingKind.EXPERIMENT,
    )
    assert sampling.kind == SamplingKind.EXPERIMENT


@pytest.mark.asyncio
async def test_list_samplings_filters_by_kind(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    from src.db.tables.sampling import SamplingKind

    await sampling_service.create_sampling(
        name="playground",
        config=minimal_sampling_config,
        kind=SamplingKind.PLAYGROUND,
    )
    experiment = await sampling_service.create_sampling(
        name="experiment",
        config={**minimal_sampling_config, "compose_grids": True},
        kind=SamplingKind.EXPERIMENT,
    )
    experiments = await sampling_service.list_samplings(kind=SamplingKind.EXPERIMENT)
    assert [item.id for item in experiments] == [experiment.id]


@pytest.mark.asyncio
async def test_enqueue_sampling_refreshes_output_path(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sampling = await sampling_service.create_sampling(name="rerun", config=minimal_sampling_config)
    first_path = sampling.output_path
    queued = await sampling_service.enqueue_sampling(sampling.id)
    assert queued.output_path is not None
    assert queued.output_path.endswith(date.today().isoformat())
    assert queued.output_path == first_path


@pytest.mark.asyncio
async def test_create_sampling_explicit_empty_lora_paths_ignores_hidden_stack(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    hidden = r"D:\loras\hidden.safetensors"
    config = {
        **minimal_sampling_config,
        "lora_paths": [hidden],
        "parameters": {
            **minimal_sampling_config["parameters"],
            "lora_path": {
                "mode": "fixed",
                "value": [
                    {"path": hidden, "trigger": "", "weight": 1.0, "bypassed": False},
                    {"path": r"D:\loras\pose.safetensors", "trigger": "", "weight": 0.6, "bypassed": False},
                ],
            },
        },
    }
    sampling = await sampling_service.create_sampling(
        name="playground",
        config=config,
        lora_paths=[],
        kind=SamplingKind.PLAYGROUND,
    )
    assert sampling.lora_paths == []
    lora_path = sampling.config["parameters"]["lora_path"]
    assert lora_path["mode"] == "fixed"
    assert lora_path["value"]["path"] is None


@pytest.mark.asyncio
async def test_create_sampling_rejects_unsupported_bf16(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    config = {**minimal_sampling_config, "mixed_precision": "bfloat16"}
    with turing_gpu(), pytest.raises(RunnableValidationError, match="bfloat16 is not supported"):
        await sampling_service.create_sampling(name="bf16", config=config)


@pytest.mark.asyncio
async def test_create_sampling_rejects_invalid_mixed_precision(
    sampling_service: SamplingService,
    minimal_sampling_config: dict,
) -> None:
    config = {**minimal_sampling_config, "mixed_precision": "fp16"}
    with pytest.raises(RunnableValidationError, match="mixed_precision"):
        await sampling_service.create_sampling(name="bad-dtype", config=config)


def test_validate_config_rejects_unsupported_bf16() -> None:
    from src.services.runnable.handlers.sampling import SamplingHandler

    with turing_gpu(), pytest.raises(RunnableValidationError, match="bfloat16 is not supported"):
        SamplingHandler().validate_config({"mixed_precision": "bfloat16"})
