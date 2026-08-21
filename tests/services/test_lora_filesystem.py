"""Tests for LoRA filesystem path and weight-file resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.db.tables.lora import Lora
from src.services.loras.paths import resolve_sample_base_dir, resolve_work_dir
from src.services.loras.service import LoraService
from src.services.loras.weights import is_checkpoint_weights, pick_weights_file


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


def test_list_samples_delegates_to_runnable_helper() -> None:
    lora = Lora(name="run", output_path="/tmp/out")
    service = LoraService(MagicMock(), MagicMock())
    with patch("src.services.loras.service.list_runnable_samples", return_value=[("preview",)]) as list_samples:
        assert service.list_samples(lora) == [("preview",)]
        list_samples.assert_called_once_with(lora)


def test_pick_weights_prefers_final_over_checkpoint(tmp_path: Path) -> None:
    work_dir = tmp_path / "demo"
    work_dir.mkdir()
    (work_dir / "demo_epoch1.safetensors").write_bytes(b"epoch")
    final = work_dir / "demo.safetensors"
    final.write_bytes(b"final")

    assert pick_weights_file(work_dir) == final


def test_pick_weights_uses_latest_checkpoint_when_no_final(tmp_path: Path) -> None:
    work_dir = tmp_path / "demo"
    work_dir.mkdir()
    (work_dir / "demo_epoch1.safetensors").write_bytes(b"1")
    latest = work_dir / "demo_epoch10.safetensors"
    latest.write_bytes(b"10")

    assert pick_weights_file(work_dir) == latest


def test_is_checkpoint_weights_detects_epoch_and_step() -> None:
    assert is_checkpoint_weights(Path("demo_epoch3.safetensors"))
    assert is_checkpoint_weights(Path("demo_step12.safetensors"))
    assert not is_checkpoint_weights(Path("demo.safetensors"))
