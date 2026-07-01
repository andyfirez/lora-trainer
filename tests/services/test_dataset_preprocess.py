"""Tests for dataset image preprocessing geometry and validation."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from src.db.tables.dataset import Dataset
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.services.datasets.exceptions import DatasetNotPreparedError, DatasetResolutionMismatchError
from src.services.datasets.preprocess import (
    BucketPreprocessConfig,
    ImagePreprocessState,
    StoredCropRecord,
    _fit_size,
    apply_bucket_crop,
    apply_crop,
    assign_bucket,
    bake_image_to_prepared,
    build_crop_meta,
    compute_preprocess_status,
    get_image_state,
    prepared_dir_path,
    recompute_preprocess_ready,
)
from src.services.datasets.training_validation import validate_dataset_for_training


def _save_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int] = (128, 64, 32)) -> None:
    Image.new("RGB", size, color).save(path)


def _square_bucket_config(resolution: int = 1024) -> BucketPreprocessConfig:
    return BucketPreprocessConfig(
        enable_bucket=False,
        resolution=resolution,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_reso_steps=64,
        bucket_no_upscale=True,
    )


def _bucket_config(resolution: int = 1024) -> BucketPreprocessConfig:
    return BucketPreprocessConfig(
        enable_bucket=True,
        resolution=resolution,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_reso_steps=256,
        bucket_no_upscale=True,
    )


def test_fit_size_landscape() -> None:
    fitted = _fit_size(3000, 2000, 1024)
    assert fitted.width == 1536
    assert fitted.height == 1024


def test_apply_crop_produces_square(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    _save_rgb(source, (2000, 1000))
    image = Image.open(source)
    result = apply_crop(image, 512, 0.5, 0.5)
    assert result.size == (512, 512)


def test_bake_image_writes_square_prepared_file(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1800, 1200))
    prepared_dir = tmp_path / ".prepared" / "1024"
    output, assignment = bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        bucket_config=_square_bucket_config(1024),
        center_x=0.5,
        center_y=0.5,
    )
    assert assignment is None
    assert output.is_file()
    with Image.open(output) as baked:
        assert baked.size == (1024, 1024)


def test_bake_image_writes_bucket_prepared_file(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1216, 918))
    prepared_dir = tmp_path / ".prepared" / "1024"
    output, assignment = bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        bucket_config=_bucket_config(1024),
        center_x=0.5,
        center_y=0.5,
    )
    assert assignment is not None
    assert output.is_file()
    with Image.open(output) as baked:
        assert baked.size == (assignment.bucket_width, assignment.bucket_height)
        assert baked.size != (1024, 1024)


def test_get_image_state_ready(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1024, 1024))
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        bucket_config=_square_bucket_config(1024),
        center_x=0.5,
        center_y=0.5,
    )
    mtime = source.stat().st_mtime
    baked_at = datetime.now(timezone.utc)
    state = get_image_state(
        filename="img.png",
        image_dir=tmp_path,
        bucket_config=_square_bucket_config(1024),
        crop_record=StoredCropRecord(
            crop_center_x=0.5,
            crop_center_y=0.5,
            source_mtime=mtime,
            baked_at=baked_at,
        ),
    )
    assert state == ImagePreprocessState.READY


def test_validate_dataset_for_training_ready(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1024, 1024))
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        bucket_config=_square_bucket_config(1024),
        center_x=0.5,
        center_y=0.5,
    )
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=1024,
        preprocess_ready=True,
    )
    validate_dataset_for_training(dataset, 1024)


def test_validate_dataset_resolution_mismatch(tmp_path: Path) -> None:
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=768,
        preprocess_ready=True,
    )
    with pytest.raises(DatasetResolutionMismatchError):
        validate_dataset_for_training(dataset, 1024)


def test_validate_dataset_bucket_mode_mismatch(tmp_path: Path) -> None:
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=1024,
        preprocess_ready=True,
        enable_bucket=True,
    )
    with pytest.raises(DatasetNotPreparedError, match="enable_bucket=false"):
        validate_dataset_for_training(dataset, 1024, enable_bucket=False)


def test_recompute_preprocess_ready(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1024, 1024))
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        bucket_config=_square_bucket_config(1024),
        center_x=0.5,
        center_y=0.5,
    )
    mtime = source.stat().st_mtime
    baked_at = datetime.now(timezone.utc)
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=1024,
        preprocess_ready=False,
    )
    crop_map = {
        "img.png": StoredCropRecord(
            crop_center_x=0.5,
            crop_center_y=0.5,
            source_mtime=mtime,
            baked_at=baked_at,
        )
    }
    status = compute_preprocess_status(dataset, crop_map)
    assert status.ready == 1
    assert recompute_preprocess_ready(dataset, crop_map) is True


def test_build_crop_meta_defaults_center(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1600, 900))
    meta = build_crop_meta(
        image_path=source,
        bucket_config=_square_bucket_config(1024),
        crop_center_x=None,
        crop_center_y=None,
        stored=None,
    )
    assert meta.crop_center_x == 0.5
    assert meta.crop_center_y == 0.5
    assert meta.state == ImagePreprocessState.NO_CROP


def test_apply_bucket_crop_matches_assignment(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1216, 918))
    image = Image.open(source)
    assignment = assign_bucket(1216, 918, resolution=1024, bucket_reso_steps=256)
    result = apply_bucket_crop(image, assignment)
    assert result.size == (assignment.bucket_width, assignment.bucket_height)


def test_validate_bucket_training_with_crops(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1216, 918))
    bucket_config = _bucket_config(1024)
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    _, assignment = bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        bucket_config=bucket_config,
        center_x=0.5,
        center_y=0.5,
    )
    assert assignment is not None
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=1024,
        preprocess_ready=True,
        enable_bucket=True,
    )
    crop = DatasetImageCrop(
        dataset_id=1,
        filename="img.png",
        crop_center_x=0.5,
        crop_center_y=0.5,
        source_mtime=source.stat().st_mtime,
        bucket_width=assignment.bucket_width,
        bucket_height=assignment.bucket_height,
        scale_to_width=assignment.scale_to_width,
        scale_to_height=assignment.scale_to_height,
        crop_x=assignment.crop_x,
        crop_y=assignment.crop_y,
    )
    validate_dataset_for_training(dataset, 1024, enable_bucket=True, crops=[crop])
