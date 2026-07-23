"""Match relocated dataset folders to stale catalog records."""

from __future__ import annotations

from src.db.tables.dataset import Dataset
from src.services.storage.relocation import (
    folder_basename,
    match_by_basename,
    unique_match,
)


def _match_by_crop_filenames(
    stale_datasets: list[Dataset],
    disk_image_filenames: frozenset[str],
    crop_filenames_by_dataset_id: dict[int, frozenset[str]],
) -> list[Dataset]:
    if not disk_image_filenames:
        return []

    matches: list[Dataset] = []
    for dataset in stale_datasets:
        if dataset.id is None:
            continue
        crop_filenames = crop_filenames_by_dataset_id.get(dataset.id)
        if not crop_filenames:
            continue
        if crop_filenames == disk_image_filenames or crop_filenames <= disk_image_filenames:
            matches.append(dataset)
    return matches


def find_relocated_dataset(
    stale_datasets: list[Dataset],
    discovered_relative_path: str,
    *,
    disk_image_filenames: frozenset[str],
    crop_filenames_by_dataset_id: dict[int, frozenset[str]],
) -> Dataset | None:
    discovered_basename = folder_basename(discovered_relative_path)
    by_basename = match_by_basename(
        stale_datasets,
        get_relative_path=lambda dataset: dataset.relative_path,
        discovered_basename=discovered_basename,
    )
    match = unique_match(by_basename)
    if match is not None:
        return match

    return unique_match(
        _match_by_crop_filenames(
            stale_datasets,
            disk_image_filenames,
            crop_filenames_by_dataset_id,
        )
    )
