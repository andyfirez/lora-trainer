"""Tests for dataset image preprocessing geometry and validation."""

from pathlib import Path

import pytest
from PIL import Image

from src.db.tables.dataset import Dataset
from src.services.datasets.exceptions import DatasetNotPreparedError, DatasetResolutionMismatchError
from src.services.datasets.preprocess import (
    ImagePreprocessState,
    _fit_size,
    apply_crop,
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


def test_fit_size_landscape() -> None:
    fitted = _fit_size(3000, 2000, 1024)
    assert fitted.width == 1536
    assert fitted.height == 1024


def test_fit_size_portrait() -> None:
    fitted = _fit_size(2000, 3000, 1024)
    assert fitted.width == 1024
    assert fitted.height == 1536


def test_apply_crop_produces_square(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    _save_rgb(source, (2000, 1000))
    image = Image.open(source)
    result = apply_crop(image, 512, 0.5, 0.5)
    assert result.size == (512, 512)


def test_bake_image_writes_prepared_file(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1800, 1200))
    prepared_dir = tmp_path / ".prepared" / "1024"
    output = bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        resolution=1024,
        center_x=0.5,
        center_y=0.5,
    )
    assert output.is_file()
    with Image.open(output) as baked:
        assert baked.size == (1024, 1024)


def test_get_image_state_ready(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1024, 1024))
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        resolution=1024,
        center_x=0.5,
        center_y=0.5,
    )
    mtime = source.stat().st_mtime
    from datetime import datetime, timezone

    baked_at = datetime.now(timezone.utc)
    state = get_image_state(
        filename="img.png",
        image_dir=tmp_path,
        resolution=1024,
        crop_mtime=mtime,
        crop_baked_at=baked_at,
    )
    assert state == ImagePreprocessState.READY


def test_validate_dataset_for_training_ready(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1024, 1024))
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        resolution=1024,
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


def test_validate_dataset_not_prepared(tmp_path: Path) -> None:
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=1024,
        preprocess_ready=False,
    )
    with pytest.raises(DatasetNotPreparedError):
        validate_dataset_for_training(dataset, 1024)


def test_recompute_preprocess_ready(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1024, 1024))
    prepared_dir = prepared_dir_path(tmp_path, 1024)
    bake_image_to_prepared(
        source_path=source,
        prepared_dir=prepared_dir,
        resolution=1024,
        center_x=0.5,
        center_y=0.5,
    )
    mtime = source.stat().st_mtime
    from datetime import datetime, timezone

    baked_at = datetime.now(timezone.utc)
    dataset = Dataset(
        id=1,
        name="demo",
        image_dir=str(tmp_path),
        target_resolution=1024,
        preprocess_ready=False,
    )
    crop_map = {"img.png": (0.5, 0.5, mtime, baked_at)}
    status = compute_preprocess_status(dataset, crop_map)
    assert status.ready == 1
    assert recompute_preprocess_ready(dataset, crop_map) is True


def test_build_crop_meta_defaults_center(tmp_path: Path) -> None:
    source = tmp_path / "img.png"
    _save_rgb(source, (1600, 900))
    meta = build_crop_meta(
        image_path=source,
        resolution=1024,
        crop_center_x=None,
        crop_center_y=None,
        crop_mtime=None,
        crop_baked_at=None,
    )
    assert meta.crop_center_x == 0.5
    assert meta.crop_center_y == 0.5
    assert meta.state == ImagePreprocessState.NO_CROP
