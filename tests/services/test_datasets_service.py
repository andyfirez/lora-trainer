from pathlib import Path

import pytest

from src.services.datasets.exceptions import DatasetDirectoryNotFoundError, DatasetNameConflictError
from src.services.datasets.service import DatasetsService


@pytest.mark.asyncio
async def test_update_dataset_name_and_image_dir(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    dataset = await datasets_service.create_dataset(name="original", image_dir=str(image_dir))
    updated = await datasets_service.update_dataset(
        dataset.id,
        name="renamed",
        image_dir=str(other_dir),
        caption_dir=None,
        description=None,
    )

    assert updated.name == "renamed"
    assert updated.image_dir == str(other_dir)


@pytest.mark.asyncio
async def test_update_dataset_name_conflict(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    await datasets_service.create_dataset(name="first", image_dir=str(image_dir))
    second = await datasets_service.create_dataset(name="second", image_dir=str(image_dir))

    with pytest.raises(DatasetNameConflictError):
        await datasets_service.update_dataset(
            second.id,
            name="first",
            image_dir=None,
            caption_dir=None,
            description=None,
        )


@pytest.mark.asyncio
async def test_update_dataset_missing_directory(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    dataset = await datasets_service.create_dataset(name="demo", image_dir=str(image_dir))

    with pytest.raises(DatasetDirectoryNotFoundError):
        await datasets_service.update_dataset(
            dataset.id,
            name=None,
            image_dir=str(tmp_path / "missing"),
            caption_dir=None,
            description=None,
        )
