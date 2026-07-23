"""Detect and remove exact duplicate images in a dataset directory."""

from dataclasses import dataclass
from pathlib import Path

from src.services.datasets.captions import (
    DEFAULT_CAPTION_EXTENSION,
    list_image_filenames,
)
from src.services.datasets.hashing import file_sha256


@dataclass(frozen=True)
class DuplicateScanResult:
    duplicate_count: int
    duplicate_filenames: tuple[str, ...]


def scan_duplicates(image_dir: Path) -> DuplicateScanResult:
    hash_to_filenames: dict[str, list[str]] = {}
    for filename in list_image_filenames(image_dir):
        path = image_dir / filename
        file_hash = file_sha256(path)
        hash_to_filenames.setdefault(file_hash, []).append(filename)

    duplicate_filenames: list[str] = []
    for filenames in hash_to_filenames.values():
        if len(filenames) <= 1:
            continue
        keep, *duplicates = sorted(filenames)
        del keep
        duplicate_filenames.extend(duplicates)

    duplicate_filenames.sort()
    return DuplicateScanResult(
        duplicate_count=len(duplicate_filenames),
        duplicate_filenames=tuple(duplicate_filenames),
    )


def remove_duplicate_files(
    image_dir: Path,
    filenames: list[str],
    caption_extension: str = DEFAULT_CAPTION_EXTENSION,
) -> int:
    removed = 0
    for filename in filenames:
        image_path = image_dir / filename
        if image_path.is_file():
            image_path.unlink()
            removed += 1
        caption_path = image_dir / f"{Path(filename).stem}{caption_extension}"
        if caption_path.is_file():
            caption_path.unlink()
    return removed
