"""Tests for LoRA path normalization helpers."""

from pathlib import Path

from src.db.migrations.lora_paths import normalize_lora_relative_path, to_lora_relative


def test_normalize_lora_relative_path_from_absolute(tmp_path: Path) -> None:
    root = tmp_path / "lora"
    work_dir = root / "characters" / "demo"
    work_dir.mkdir(parents=True)

    assert normalize_lora_relative_path(str(work_dir.resolve()), root) == "characters/demo"


def test_to_lora_relative_returns_none_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "lora"
    root.mkdir(exist_ok=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    assert to_lora_relative(root, outside) is None
