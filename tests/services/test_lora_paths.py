"""Tests for LoRA filesystem path resolution."""

from pathlib import Path

import pytest

from src.db.tables.lora import Lora
from src.services.loras.paths import (
    build_training_loss_log_path,
    resolve_sample_base_dir,
    resolve_training_work_dir_from_config,
    resolve_work_dir,
)
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import build_loss_log_path


def test_build_loss_log_path_uses_resolved_work_dir(storage_roots) -> None:
    config = TrainConfig(output_dir="", lora_name="path-check")
    work_dir = resolve_training_work_dir_from_config(config)
    assert build_loss_log_path(config) == build_training_loss_log_path(work_dir)


def test_resolve_sample_base_dir_prefers_output_path(tmp_path: Path) -> None:
    output = tmp_path / "active-run"
    output.mkdir()
    lora = Lora(name="run", relative_path="", output_path=str(output))
    assert resolve_sample_base_dir(lora) == output


def test_resolve_sample_base_dir_falls_back_to_work_dir(storage_roots) -> None:
    work_dir = storage_roots["lora"] / "completed"
    work_dir.mkdir()
    lora = Lora(name="done", relative_path="completed", output_path=None)
    assert resolve_sample_base_dir(lora) == resolve_work_dir(lora)


@pytest.mark.asyncio
async def test_list_samples_uses_output_path_during_run(
    lora_service,
    minimal_training_yaml: str,
    storage_roots,
) -> None:
    lora = await lora_service.create_lora(name="sample-run", config_yaml=minimal_training_yaml)
    output_dir = storage_roots["lora"] / "sample-run-active"
    output_dir.mkdir()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir()
    (samples_dir / "preview.png").write_bytes(b"fake-png")

    lora.output_path = str(output_dir)
    lora.relative_path = ""

    found = lora_service.list_samples(lora)
    assert len(found) == 1
    assert found[0][0].name == "preview.png"
