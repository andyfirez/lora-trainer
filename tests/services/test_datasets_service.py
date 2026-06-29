from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.db.tables.dataset import Dataset
from src.services.datasets.exceptions import DatasetDirectoryNotFoundError, DatasetNameConflictError
from src.services.datasets.preprocess import ImagePreprocessState, prepared_dir_path
from src.services.datasets.service import DatasetsService


def _write_test_image(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 100, 100)).save(path)


async def _create_dataset_with_resolution(
    datasets_service: DatasetsService,
    image_dir: Path,
    *,
    name: str = "demo",
    resolution: int = 1024,
) -> Dataset:
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = await datasets_service.create_dataset(name=name, image_dir=str(image_dir))
    return await datasets_service.update_dataset(
        dataset.id,
        name=None,
        image_dir=None,
        caption_dir=None,
        description=None,
        target_resolution=resolution,
        update_target_resolution=True,
    )


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


@pytest.mark.asyncio
async def test_bake_all_creates_default_crop_and_bakes(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    _write_test_image(image_dir / "img.png")
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)

    baked = await datasets_service.bake_all(dataset)

    assert baked == 1
    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
    assert dataset.preprocess_ready is True
    prepared = prepared_dir_path(image_dir, 1024) / "img.png"
    assert prepared.is_file()
    with Image.open(prepared) as img:
        assert img.size == (1024, 1024)


@pytest.mark.asyncio
async def test_bake_all_rebakes_stale_image(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    image_path = image_dir / "img.png"
    _write_test_image(image_path)
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)
    await datasets_service.save_crop(dataset, "img.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "img.png")

    _write_test_image(image_path, size=(900, 700))
    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]

    baked = await datasets_service.bake_all(dataset)

    assert baked == 1
    crop = await datasets_service.get_crop_meta(dataset, "img.png")
    assert crop.crop_center_x == pytest.approx(0.5)
    assert crop.crop_center_y == pytest.approx(0.5)
    assert crop.state == ImagePreprocessState.READY


@pytest.mark.asyncio
async def test_bake_all_skips_ready_images(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    _write_test_image(image_dir / "img.png")
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)
    await datasets_service.bake_all(dataset)

    baked = await datasets_service.bake_all(dataset)

    assert baked == 0
    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
    assert dataset.preprocess_ready is True


@pytest.mark.asyncio
async def test_update_tags_invalidates_te_cache(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    _write_test_image(image_dir / "img.png")
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)
    await datasets_service.bake_all(dataset)

    prepared_dir = prepared_dir_path(image_dir, 1024)
    cache_path = prepared_dir / "img_te.npz"
    np.savez(
        cache_path,
        prompt_embeds=np.zeros((1, 77, 2048), dtype=np.float32),
        pooled_prompt_embeds=np.zeros((1, 1280), dtype=np.float32),
    )

    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
    datasets_service.update_tags(dataset, "img.png", ["solo", "1girl"])

    assert not cache_path.is_file()
