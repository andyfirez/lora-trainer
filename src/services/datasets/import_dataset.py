"""Copy external folders into managed dataset storage."""

import shutil
from pathlib import Path

from src.storage.paths import StorageKind, StoragePaths


def copy_dataset_import(source_dir: Path, relative_path: str) -> Path:
    """Copy an external directory into datasets_root at relative_path."""
    source = source_dir.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory not found: {source}")

    validated = StoragePaths.validate_relative_path(StorageKind.DATASETS, relative_path)
    dest = StoragePaths.resolve_dataset_path(validated)
    if dest.exists():
        raise ValueError(f"Destination already exists: {validated}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, symlinks=False)
    return dest
