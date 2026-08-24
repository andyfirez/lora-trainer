"""Remove dataset directories from disk."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.storage.paths import StoragePaths

logger = logging.getLogger(__name__)


def remove_dataset_directory(relative_path: str) -> None:
    """Delete the on-disk dataset folder if it exists and passes safety checks."""
    try:
        path = StoragePaths.resolve_dataset_path(relative_path)
    except (ValueError, OSError) as exc:
        logger.warning("Skipping dataset directory removal for %r: %s", relative_path, exc)
        return

    resolved = path.resolve()
    if not resolved.is_dir():
        return

    if resolved.parent == resolved:
        logger.warning("Refusing to remove filesystem root: %s", resolved)
        return

    datasets_root = StoragePaths.datasets_root().resolve()
    if resolved == datasets_root:
        logger.warning("Refusing to remove datasets root: %s", resolved)
        return

    if not Path(relative_path).is_absolute() and not StoragePaths._is_under_root(resolved, datasets_root):
        logger.warning("Refusing to remove relative path outside datasets root: %s", resolved)
        return

    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        logger.warning("Failed to remove dataset directory %s: %s", resolved, exc)
