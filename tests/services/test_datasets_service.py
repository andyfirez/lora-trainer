from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from src.db.tables.dataset import Dataset
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetImageNotFoundError,
    DatasetNameConflictError,
)
from src.services.datasets.preprocess import ImagePreprocessState, prepared_dir_path
from src.services.datasets.service import DatasetsService
from src.services.tagging.manager import TaggingStatus, TaggingTaskState
from src.tagger.config import TaggingConfig


def _write_test_image(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 100, 100)).save(path)


async def _create_dataset_with_resolution(
    datasets_service: DatasetsService,
    image_dir: Path,
    *,
    name: str = "demo",
    relative_path: str = "images",
    resolution: int = 1024,
) -> Dataset:
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = await datasets_service.create_dataset(name=name, relative_path=relative_path)
    return await datasets_service.update_dataset(
        dataset.id,
        target_resolution=resolution,
    )


@pytest.mark.asyncio
async def test_update_dataset_absolute_path_normalized(
    storage_roots,
    datasets_service: DatasetsService,
) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    other_dir = storage_roots["datasets"] / "other"
    other_dir.mkdir()

    dataset = await datasets_service.create_dataset(name="original", relative_path="images")
    updated = await datasets_service.update_dataset(
        dataset.id,
        relative_path=str(other_dir),
    )

    assert updated.relative_path == "other"


@pytest.mark.asyncio
async def test_update_dataset_name_and_image_dir(
    storage_roots,
    datasets_service: DatasetsService,
) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    other_dir = storage_roots["datasets"] / "other"
    other_dir.mkdir()

    dataset = await datasets_service.create_dataset(name="original", relative_path="images")
    updated = await datasets_service.update_dataset(
        dataset.id,
        name="renamed",
        relative_path="other",
    )

    assert updated.name == "renamed"
    assert updated.relative_path == "other"


@pytest.mark.asyncio
async def test_update_dataset_name_conflict(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()

    await datasets_service.create_dataset(name="first", relative_path="images")
    second = await datasets_service.create_dataset(name="second", relative_path="images")

    with pytest.raises(DatasetNameConflictError):
        await datasets_service.update_dataset(
            second.id,
            name="first",
        )


@pytest.mark.asyncio
async def test_update_dataset_missing_directory(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()

    dataset = await datasets_service.create_dataset(name="demo", relative_path="images")

    with pytest.raises(DatasetDirectoryNotFoundError):
        await datasets_service.update_dataset(
            dataset.id,
            relative_path="missing",
        )


@pytest.mark.asyncio
async def test_bake_all_creates_default_crop_and_bakes(
    storage_roots,
    datasets_service: DatasetsService,
) -> None:
    image_dir = storage_roots["datasets"] / "images"
    _write_test_image(image_dir / "img.png")
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)

    baked = await datasets_service.bake_all(dataset)

    assert baked == 1
    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
    assert dataset.preprocess_ready is True
    prepared = prepared_dir_path(image_dir, 1024) / "img.jpg"
    assert prepared.is_file()
    with Image.open(prepared) as img:
        assert img.size == (1024, 1024)


@pytest.mark.asyncio
async def test_bake_all_rebakes_stale_image(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
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
async def test_bake_all_skips_ready_images(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
    _write_test_image(image_dir / "img.png")
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)
    await datasets_service.bake_all(dataset)

    baked = await datasets_service.bake_all(dataset)

    assert baked == 0
    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
    assert dataset.preprocess_ready is True


@pytest.mark.asyncio
async def test_update_tags_invalidates_te_cache(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
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


@pytest.mark.asyncio
async def test_delete_image_removes_files_and_crop(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
    _write_test_image(image_dir / "keep.png")
    _write_test_image(image_dir / "remove.png")
    (image_dir / "remove.txt").write_text("solo, 1girl", encoding="utf-8")
    dataset = await _create_dataset_with_resolution(datasets_service, image_dir)
    await datasets_service.bake_all(dataset)

    prepared = prepared_dir_path(image_dir, 1024) / "remove.jpg"
    assert prepared.is_file()

    await datasets_service.delete_image(dataset, "remove.png")

    assert (image_dir / "keep.png").is_file()
    assert not (image_dir / "remove.png").is_file()
    assert not (image_dir / "remove.txt").is_file()
    assert not prepared.is_file()

    dataset = await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
    assert dataset.preprocess_ready is True

    with pytest.raises(DatasetImageNotFoundError):
        await datasets_service.delete_image(dataset, "remove.png")


@pytest.mark.asyncio
async def test_start_autotag_builds_config_and_delegates(
    storage_roots,
    datasets_service: DatasetsService,
) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    dataset = await datasets_service.create_dataset(name="tags", relative_path="images")
    expected = TaggingTaskState(status=TaggingStatus.RUNNING, current=0, total=1, message="")
    datasets_service._tags._tagging = MagicMock()
    datasets_service._tags._tagging.start.return_value = expected

    state = datasets_service.start_autotag(dataset, threshold=0.4, filenames=["cat.png"])

    assert state is expected
    datasets_service._tags._tagging.start.assert_called_once()
    kwargs = datasets_service._tags._tagging.start.call_args
    assert kwargs.args[0] == dataset.id
    config = kwargs.kwargs["config"]
    assert isinstance(config, TaggingConfig)
    assert config.threshold == 0.4
    assert config.filenames == ["cat.png"]


@pytest.mark.asyncio
async def test_get_autotag_status_returns_idle_when_missing(
    storage_roots,
    datasets_service: DatasetsService,
) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    dataset = await datasets_service.create_dataset(name="idle", relative_path="images")
    datasets_service._tags._tagging = MagicMock()
    datasets_service._tags._tagging.get_status.return_value = None

    state = datasets_service.get_autotag_status(dataset)

    assert state.status == TaggingStatus.IDLE
    assert state.current == 0
    assert state.total == 0
