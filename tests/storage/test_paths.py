"""Tests for managed storage path resolution."""

import pytest
from src.storage.paths import StorageKind, StoragePaths


def test_resolve_dataset_path_relative(storage_roots) -> None:
    target = storage_roots["datasets"] / "anime" / "girl"
    target.mkdir(parents=True)
    resolved = StoragePaths.resolve_dataset_path("anime/girl")
    assert resolved == target.resolve()


def test_resolve_training_work_dir(storage_roots) -> None:
    work = StoragePaths.resolve_training_work_dir("characters", "my_lora_j42")
    assert work == (storage_roots["lora"] / "characters" / "my_lora_j42").resolve()


def test_resolve_training_work_dir_absolute_output_dir(storage_roots) -> None:
    output_dir = storage_roots["lora"] / "characters"
    output_dir.mkdir()
    work = StoragePaths.resolve_training_work_dir(str(output_dir), "my_lora_j42")
    assert work == (output_dir / "my_lora_j42").resolve()


def test_normalize_input_path_absolute_under_root(storage_roots) -> None:
    target = storage_roots["datasets"] / "anime" / "girl"
    target.mkdir(parents=True)
    normalized = StoragePaths.normalize_input_path(StorageKind.DATASETS, str(target))
    assert normalized == "anime/girl"


def test_normalize_input_path_rejects_outside_root(storage_roots, tmp_path) -> None:
    outside = tmp_path / "outside-managed"
    outside.mkdir()
    with pytest.raises(ValueError, match="datasets"):
        StoragePaths.normalize_input_path(StorageKind.DATASETS, str(outside))


def test_dataset_dir_exists_false_outside_root(storage_roots) -> None:
    assert StoragePaths.dataset_dir_exists("/tmp/not-managed") is False


def test_validate_relative_path_rejects_parent_segments(storage_roots) -> None:
    with pytest.raises(ValueError, match="parent"):
        StoragePaths.validate_relative_path(StorageKind.DATASETS, "../escape")
