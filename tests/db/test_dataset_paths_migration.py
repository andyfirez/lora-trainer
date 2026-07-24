"""Tests for dataset path normalization migration helpers."""

from pathlib import Path

from src.db.migrations.dataset_paths import normalize_dataset_relative_path


def test_normalize_absolute_path_under_root(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset_dir = root / "Winx" / "Winx_Chimera"
    dataset_dir.mkdir(parents=True)

    result = normalize_dataset_relative_path(str(dataset_dir.resolve()), root.resolve())
    assert result == "Winx/Winx_Chimera"


def test_normalize_relative_path_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir(exist_ok=True)

    assert normalize_dataset_relative_path("Characters/Melanie", root) == "Characters/Melanie"


def test_normalize_absolute_outside_root_left_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir(exist_ok=True)
    outside = tmp_path / "elsewhere" / "dataset"
    outside.mkdir(parents=True)

    absolute = str(outside.resolve())
    assert normalize_dataset_relative_path(absolute, root) == absolute
