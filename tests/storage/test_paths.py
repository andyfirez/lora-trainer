"""Tests for managed storage path resolution."""

from pathlib import Path

import pytest

from src.settings.app_settings import settings
from src.storage.paths import StorageKind, StoragePaths


@pytest.fixture
def storage_roots(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    base_models = tmp_path / "base-models"
    lora = tmp_path / "lora"
    for path in (datasets, base_models, lora):
        path.mkdir()
    settings.storage = settings.storage.model_copy(
        update={
            "datasets_root": str(datasets),
            "base_models_root": str(base_models),
            "lora_root": str(lora),
        }
    )
    return {"datasets": datasets, "base_models": base_models, "lora": lora}


def test_resolve_dataset_path_relative(storage_roots) -> None:
    target = storage_roots["datasets"] / "anime" / "girl"
    target.mkdir(parents=True)
    resolved = StoragePaths.resolve_dataset_path("anime/girl")
    assert resolved == target.resolve()


def test_resolve_training_work_dir(storage_roots) -> None:
    work = StoragePaths.resolve_training_work_dir("characters", "my_lora_j42")
    assert work == (storage_roots["lora"] / "characters" / "my_lora_j42").resolve()


def test_dataset_dir_exists_false_outside_root(storage_roots) -> None:
    assert StoragePaths.dataset_dir_exists("/tmp/not-managed") is False


def test_validate_relative_path_rejects_parent_segments(storage_roots) -> None:
    with pytest.raises(ValueError, match="parent"):
        StoragePaths.validate_relative_path(StorageKind.DATASETS, "../escape")
