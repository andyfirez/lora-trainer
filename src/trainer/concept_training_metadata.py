"""Resolve per-image training metadata from dataset crop records."""

import statistics
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.services.datasets.captions import list_image_filenames
from src.services.datasets.paths import dataset_image_dir
from src.services.datasets.exceptions import DatasetNotFoundError
from src.trainer.sdxl.buckets import assignment_from_stored, compute_add_time_ids


@dataclass(frozen=True)
class ImageTrainingMeta:
    filename: str
    add_time_ids: tuple[int, int, int, int, int, int]
    bucket_width: int
    bucket_height: int


@dataclass(frozen=True)
class ConceptTrainingMetadata:
    dataset_id: int
    enable_bucket: bool
    by_filename: dict[str, ImageTrainingMeta]


async def resolve_concept_training_metadata(
    dataset_ids: list[int],
    dataset_repo: DatasetRepository,
    crop_repo: DatasetImageCropRepository,
) -> dict[int, ConceptTrainingMetadata]:
    result: dict[int, ConceptTrainingMetadata] = {}
    unique_ids = list(dict.fromkeys(dataset_ids))
    for dataset_id in unique_ids:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        if dataset.target_resolution is None:
            raise ValueError(f"Dataset with id={dataset_id} has no target_resolution")
        crops = await crop_repo.list_by_dataset(dataset_id)
        image_dir = dataset_image_dir(dataset)
        disk_filenames = set(list_image_filenames(image_dir))
        by_filename: dict[str, ImageTrainingMeta] = {}
        resolution = dataset.target_resolution
        for crop in crops:
            if crop.filename not in disk_filenames:
                continue
            if not dataset.enable_bucket:
                add_time_ids = (resolution, resolution, 0, 0, resolution, resolution)
                by_filename[crop.filename] = ImageTrainingMeta(
                    filename=crop.filename,
                    add_time_ids=add_time_ids,
                    bucket_width=resolution,
                    bucket_height=resolution,
                )
                continue
            if crop.bucket_width is None or crop.bucket_height is None:
                raise ValueError(
                    f"Dataset {dataset_id}: missing bucket metadata for {crop.filename}; rebake with bucketing"
                )
            if crop.scale_to_width is None or crop.scale_to_height is None:
                raise ValueError(
                    f"Dataset {dataset_id}: missing scale metadata for {crop.filename}; rebake with bucketing"
                )
            image_path = image_dir / crop.filename
            with Image.open(image_path) as img:
                source_width, source_height = img.size
            assignment = assignment_from_stored(
                source_width=source_width,
                source_height=source_height,
                bucket_width=crop.bucket_width,
                bucket_height=crop.bucket_height,
                scale_to_width=crop.scale_to_width,
                scale_to_height=crop.scale_to_height,
                crop_x=crop.crop_x,
                crop_y=crop.crop_y,
            )
            by_filename[crop.filename] = ImageTrainingMeta(
                filename=crop.filename,
                add_time_ids=compute_add_time_ids(assignment),
                bucket_width=crop.bucket_width,
                bucket_height=crop.bucket_height,
            )
        result[dataset_id] = ConceptTrainingMetadata(
            dataset_id=dataset_id,
            enable_bucket=dataset.enable_bucket,
            by_filename=by_filename,
        )
    return result


def resolve_reference_add_time_ids(
    metadata_by_dataset: dict[int, ConceptTrainingMetadata],
    dataset_ids: list[int],
    width: int,
    height: int,
) -> tuple[float, float, float, float, float, float] | None:
    """Median train-time add_time_ids for images whose bucket matches (width, height).

    Returns None when no matching bucket data exists (falls back to (H,W,0,0,H,W)).
    """
    matching: list[tuple[int, int, int, int, int, int]] = []
    for dataset_id in dict.fromkeys(dataset_ids):
        concept_meta = metadata_by_dataset.get(dataset_id)
        if concept_meta is None:
            continue
        for image_meta in concept_meta.by_filename.values():
            if image_meta.bucket_width == width and image_meta.bucket_height == height:
                matching.append(image_meta.add_time_ids)
    if not matching:
        return None
    source_heights = [ids[0] for ids in matching]
    source_widths = [ids[1] for ids in matching]
    crop_tops = [ids[2] for ids in matching]
    crop_lefts = [ids[3] for ids in matching]
    return (
        float(statistics.median(source_heights)),
        float(statistics.median(source_widths)),
        float(statistics.median(crop_tops)),
        float(statistics.median(crop_lefts)),
        float(height),
        float(width),
    )
