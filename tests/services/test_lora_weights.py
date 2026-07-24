"""Tests for LoRA weight file selection."""

from pathlib import Path

from src.services.loras.weights import is_checkpoint_weights, pick_weights_file


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


def test_is_checkpoint_weights_detects_epoch_and_step(tmp_path: Path) -> None:
    assert is_checkpoint_weights(Path("demo_epoch3.safetensors"))
    assert is_checkpoint_weights(Path("demo_step12.safetensors"))
    assert not is_checkpoint_weights(Path("demo.safetensors"))
