"""Training-time validation for prepared datasets."""

from pathlib import Path

from PIL import Image

from src.db.tables.dataset import Dataset
from src.services.datasets.captions import list_image_filenames
from src.services.datasets.exceptions import (
    DatasetNotPreparedError,
    DatasetResolutionMismatchError,
)
from src.services.datasets.preprocess import prepared_dir_path


def validate_dataset_for_training(dataset: Dataset, resolution: int) -> None:
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

    missing: list[str] = []
    invalid_size: list[str] = []
    for filename in filenames:
        prepared_path = prepared_dir / filename
        if not prepared_path.is_file():
            alt_png = prepared_dir / f"{Path(filename).stem}.png"
            if alt_png.is_file():
                prepared_path = alt_png
            else:
                missing.append(filename)
                continue
        try:
            with Image.open(prepared_path) as img:
                if img.size != (resolution, resolution):
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
            f"{len(invalid_size)} prepared image(s) have wrong size (expected {resolution}x{resolution})",
        )

    prepared_files = [
        p.name
        for p in prepared_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    if len(prepared_files) != len(filenames):
        raise DatasetNotPreparedError(
            dataset.id,
            dataset.name,
            f"Prepared image count ({len(prepared_files)}) != original count ({len(filenames)})",
        )
