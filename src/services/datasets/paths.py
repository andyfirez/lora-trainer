"""Dataset path helpers for managed storage."""

from pathlib import Path

from src.db.tables.dataset import Dataset
from src.storage.paths import StoragePaths


def dataset_image_dir(dataset: Dataset) -> Path:
    return StoragePaths.resolve_dataset_path(dataset.relative_path)
