"""Tests for sampling LoRA path resolution via inference root."""

from __future__ import annotations

from pathlib import Path

from src.storage.config_paths import resolve_sampling_lora_file


def test_resolve_sampling_lora_file_relative(storage_roots) -> None:
    inference_root: Path = storage_roots["inference_lora"]
    file_path = inference_root / "sample.safetensors"
    file_path.write_bytes(b"x")

    resolved = resolve_sampling_lora_file("sample.safetensors")
    assert resolved == file_path.resolve()


def test_resolve_sampling_lora_file_absolute(storage_roots) -> None:
    inference_root: Path = storage_roots["inference_lora"]
    file_path = inference_root / "abs.safetensors"
    file_path.write_bytes(b"x")

    resolved = resolve_sampling_lora_file(str(file_path))
    assert resolved == file_path.resolve()
