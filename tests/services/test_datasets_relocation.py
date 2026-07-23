"""Tests for dataset catalog relocation on filesystem moves."""

from pathlib import Path

import pytest
from PIL import Image

from src.db.tables.dataset import Dataset
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.services.datasets.service import DatasetsService


def _write_test_image(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 100, 100)).save(path)


@pytest.mark.asyncio
async def test_sync_relocates_dataset_when_moved_to_subfolder(
    datasets_service: DatasetsService,
    storage_roots,
) -> None:
    image_dir = storage_roots["datasets"] / "photos"
    image_dir.mkdir()
    _write_test_image(image_dir / "img.png")

    dataset = await datasets_service.create_dataset(name="photos", relative_path="photos")
    dataset_id = dataset.id

    moved_dir = storage_roots["datasets"] / "anime" / "photos"
    moved_dir.parent.mkdir(parents=True, exist_ok=True)
    image_dir.rename(moved_dir)

    datasets = await datasets_service.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].id == dataset_id
    assert datasets[0].name == "photos"
    assert datasets[0].relative_path == "anime/photos"


@pytest.mark.asyncio
async def test_sync_relocates_dataset_when_folder_renamed_with_crops(
    datasets_service: DatasetsService,
    session,
    storage_roots,
) -> None:
    image_dir = storage_roots["datasets"] / "old_name"
    image_dir.mkdir()
    image_path = image_dir / "img.png"
    _write_test_image(image_path)

    dataset = await datasets_service.create_dataset(name="old_name", relative_path="old_name")
    session.add(
        DatasetImageCrop(
            dataset_id=dataset.id,
            filename="img.png",
            crop_center_x=0.5,
            crop_center_y=0.5,
            source_mtime=image_path.stat().st_mtime,
        )
    )
    await session.commit()
    dataset_id = dataset.id

    moved_dir = storage_roots["datasets"] / "folder" / "new_name"
    moved_dir.parent.mkdir(parents=True, exist_ok=True)
    image_dir.rename(moved_dir)

    datasets = await datasets_service.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].id == dataset_id
    assert datasets[0].name == "old_name"
    assert datasets[0].relative_path == "folder/new_name"


@pytest.mark.asyncio
async def test_sync_creates_new_dataset_when_no_stale_match(
    datasets_service: DatasetsService,
    storage_roots,
) -> None:
    image_dir = storage_roots["datasets"] / "brand_new"
    image_dir.mkdir()
    _write_test_image(image_dir / "img.png")

    datasets = await datasets_service.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].relative_path == "brand_new"
    assert datasets[0].name == "brand_new"


@pytest.mark.asyncio
async def test_sync_does_not_merge_ambiguous_stale_datasets(
    datasets_service: DatasetsService,
    session,
    storage_roots,
) -> None:
    session.add_all(
        [
            Dataset(name="alpha-photos", relative_path="alpha/photos"),
            Dataset(name="beta-photos", relative_path="beta/photos"),
        ]
    )
    await session.commit()

    image_dir = storage_roots["datasets"] / "gamma" / "photos"
    image_dir.mkdir(parents=True)
    _write_test_image(image_dir / "img.png")

    datasets = await datasets_service.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].relative_path == "gamma/photos"
    assert datasets[0].name == "photos"
