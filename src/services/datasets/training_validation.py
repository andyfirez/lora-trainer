"""Training-time validation for prepared datasets."""

from pathlib import Path

from PIL import Image

from src.db.tables.dataset import Dataset
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.services.datasets.captions import list_image_filenames
from src.services.datasets.exceptions import (
    DatasetNotPreparedError,
    DatasetResolutionMismatchError,
)
from src.services.datasets.preprocess import prepared_dir_path, resolve_prepared_path


def validate_dataset_for_training(
    dataset: Dataset,
    resolution: int,
    *,
    enable_bucket: bool = False,
    crops: list[DatasetImageCrop] | None = None,
) -> None:
    if dataset.target_resolution is None:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            "Dataset has no target_resolution; prepare images in the dataset UI first",
        )
    if dataset.target_resolution != resolution:
        raise DatasetResolutionMismatchError(
            dataset.id,
            dataset.name,
            dataset.target_resolution,
            resolution,
        )
    if not dataset.preprocess_ready:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            "Dataset preprocessing is not complete; crop and bake all images first",
        )
    if enable_bucket and not dataset.enable_bucket:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            "Training has enable_bucket=true but dataset was prepared without bucketing",
        )
    if not enable_bucket and dataset.enable_bucket:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            "Dataset was prepared with bucketing but training has enable_bucket=false",
        )

    image_dir = Path(dataset.image_dir)
    if not image_dir.is_dir():
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            f"Image directory not found: {dataset.image_dir}",
        )

    filenames = list_image_filenames(image_dir)
    if not filenames:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            "Dataset has no images",
        )

    prepared_dir = prepared_dir_path(dataset.image_dir, dataset.target_resolution)
    if not prepared_dir.is_dir():
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            f"Prepared directory not found: {prepared_dir}",
        )

    crop_by_filename = {crop.filename: crop for crop in (crops or [])}
    if crops is None:
        return

    missing: list[str] = []
    invalid_size: list[str] = []
    for filename in filenames:
        prepared_path = resolve_prepared_path(prepared_dir, filename)
        if prepared_path is None:
            missing.append(filename)
            continue
        crop = crop_by_filename.get(filename)
        if enable_bucket:
            if crop is None or crop.bucket_width is None or crop.bucket_height is None:
                invalid_size.append(filename)
                continue
            expected = (crop.bucket_width, crop.bucket_height)
        else:
            expected = (resolution, resolution)
        try:
            with Image.open(prepared_path) as img:
                if img.size != expected:
                    invalid_size.append(filename)
        except OSError:
            invalid_size.append(filename)

    if missing:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            f"{len(missing)} prepared image(s) missing (e.g. {missing[0]})",
        )
    if invalid_size:
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            f"{len(invalid_size)} prepared image(s) have wrong size",
        )

    prepared_count = sum(
        1 for filename in filenames if resolve_prepared_path(prepared_dir, filename) is not None
    )
    if prepared_count != len(filenames):
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            f"Prepared image count ({prepared_count}) != original count ({len(filenames)})",
        )
