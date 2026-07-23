"""Reconcile dataset DB metadata with the filesystem (disk is source of truth)."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.tables.dataset import Dataset
from src.services.datasets.paths import dataset_image_dir, dataset_image_dir_str
from src.services.datasets.formats import IMAGE_EXTENSIONS, PREPARED_EXTENSION
from src.services.datasets.hashing import file_sha256
from src.services.datasets.preprocess import (
    invalidate_latent_cache_for_prepared,
    prepared_dir_path,
    resolve_prepared_path,
)
from src.services.datasets.training_cache import invalidate_te_cache_for_image

PurgeArtifacts = Callable[[Dataset, str], Awaitable[None]]


@dataclass
class DatasetReconcileResult:
    removed_orphans: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    removed_prepared_orphans: list[str] = field(default_factory=list)
    preprocess_ready_updated: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.removed_orphans or self.renamed or self.removed_prepared_orphans)


def _disk_stems(image_dir: Path, filenames: list[str]) -> set[str]:
    return {Path(filename).stem for filename in filenames}


def _remove_prepared_artifacts_for_stem(
    *,
    image_dir: str | Path,
    target_resolution: int,
    stem: str,
    filename_hint: str | None = None,
) -> list[str]:
    removed: list[str] = []
    prepared_dir = prepared_dir_path(image_dir, target_resolution)
    if not prepared_dir.is_dir():
        return removed

    candidates: list[Path] = []
    if filename_hint is not None:
        resolved = resolve_prepared_path(prepared_dir, filename_hint)
        if resolved is not None:
            candidates.append(resolved)

    for ext in IMAGE_EXTENSIONS:
        candidates.append(prepared_dir / f"{stem}{ext}")
    candidates.append(prepared_dir / f"{stem}{PREPARED_EXTENSION}")
    candidates.append(prepared_dir / f"{stem}_sdxl.npz")
    candidates.append(prepared_dir / f"{stem}_te.npz")

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS or path.name.endswith("_sdxl.npz"):
            invalidate_latent_cache_for_prepared(path)
        path.unlink(missing_ok=True)
        removed.append(path.name)

    if filename_hint is not None:
        invalidate_te_cache_for_image(image_dir, filename_hint, target_resolution)

    return removed


def _remove_orphan_prepared_files(
    *,
    image_dir: Path,
    target_resolution: int,
    disk_filenames: list[str],
) -> list[str]:
    prepared_dir = prepared_dir_path(image_dir, target_resolution)
    if not prepared_dir.is_dir():
        return []

    valid_stems = _disk_stems(image_dir, disk_filenames)
    removed: list[str] = []
    for path in prepared_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_sdxl.npz"):
            stem = name[: -len("_sdxl.npz")]
        elif name.endswith("_te.npz"):
            stem = name[: -len("_te.npz")]
        else:
            stem = path.stem
        if stem in valid_stems:
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS or name.endswith("_sdxl.npz"):
            invalidate_latent_cache_for_prepared(path)
        path.unlink(missing_ok=True)
        removed.append(name)
    return removed


async def reconcile_dataset_records(
    dataset: Dataset,
    crop_repo: DatasetImageCropRepository,
    *,
    purge_artifacts: PurgeArtifacts,
) -> DatasetReconcileResult:
    """Sync crop records and prepared artifacts with files on disk."""
    result = DatasetReconcileResult()
    image_dir = dataset_image_dir(dataset)
    if not image_dir.is_dir():
        return result

    disk_filenames = list_image_filenames(image_dir)
    disk_set = set(disk_filenames)
    crops = list(await crop_repo.list_by_dataset(dataset.id))  # type: ignore[arg-type]
    crop_by_filename = {crop.filename: crop for crop in crops}

    hash_by_disk_filename: dict[str, str] = {
        filename: file_sha256(image_dir / filename) for filename in disk_filenames
    }
    now = datetime.now(timezone.utc)
    metadata_updated = False

    for filename in disk_filenames:
        crop = crop_by_filename.get(filename)
        if crop is None:
            continue
        content_hash = hash_by_disk_filename[filename]
        if crop.content_hash != content_hash:
            crop.content_hash = content_hash
            crop.updated_at = now
            crop_repo._session.add(crop)
            metadata_updated = True

    orphan_crops = [crop for crop in crops if crop.filename not in disk_set]
    new_without_crop = [filename for filename in disk_filenames if filename not in crop_by_filename]

    matched_orphans: set[int] = set()
    matched_new: set[str] = set()

    hash_to_new_files: dict[str, list[str]] = {}
    for filename in new_without_crop:
        file_hash = hash_by_disk_filename[filename]
        hash_to_new_files.setdefault(file_hash, []).append(filename)

    if dataset.target_resolution is not None:
        for crop in orphan_crops:
            if crop.id is not None and crop.id in matched_orphans:
                continue
            if not crop.content_hash:
                continue
            candidates = [
                filename
                for filename in hash_to_new_files.get(crop.content_hash, [])
                if filename not in matched_new
            ]
            if len(candidates) != 1:
                continue
            new_filename = candidates[0]
            old_filename = crop.filename
            crop.filename = new_filename
            crop.content_hash = hash_by_disk_filename[new_filename]
            crop.baked_at = None
            crop.updated_at = now
            crop_repo._session.add(crop)
            _remove_prepared_artifacts_for_stem(
                image_dir=dataset_image_dir_str(dataset),
                target_resolution=dataset.target_resolution,
                stem=Path(old_filename).stem,
                filename_hint=old_filename,
            )
            matched_orphans.add(crop.id)  # type: ignore[arg-type]
            matched_new.add(new_filename)
            result.renamed.append((old_filename, new_filename))
            metadata_updated = True

    remaining_orphans = [
        crop
        for crop in orphan_crops
        if crop.id not in matched_orphans  # type: ignore[operator]
    ]
    for crop in remaining_orphans:
        await purge_artifacts(dataset, crop.filename)
        result.removed_orphans.append(crop.filename)

    if dataset.target_resolution is not None:
        result.removed_prepared_orphans = _remove_orphan_prepared_files(
            image_dir=image_dir,
            target_resolution=dataset.target_resolution,
            disk_filenames=disk_filenames,
        )

    if metadata_updated or result.changed:
        await crop_repo._session.flush()

    return result
