from pathlib import Path

import pytest

from src.trainer.sdxl.model_loader import is_checkpoint_file


def test_is_checkpoint_file_for_safetensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")
    assert is_checkpoint_file(checkpoint) is True


def test_is_checkpoint_file_for_ckpt(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"")
    assert is_checkpoint_file(checkpoint) is True


def test_is_checkpoint_file_for_directory(tmp_path: Path) -> None:
    assert is_checkpoint_file(tmp_path / "stabilityai" / "stable-diffusion-xl-base-1.0") is False


def test_is_checkpoint_file_for_other_extension(tmp_path: Path) -> None:
    other = tmp_path / "model.pt"
    other.write_bytes(b"")
    assert is_checkpoint_file(other) is False
