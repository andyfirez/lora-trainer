"""Tests for dataset filesystem reconcile."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image
from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.services.datasets.preprocess import (
    BucketPreprocessConfig,
    ImagePreprocessState,
    StoredCropRecord,
    get_image_state,
    has_complete_bucket_metadata,
    prepared_dir_path,
)
from src.services.datasets.service import DatasetsService
from src.trainer.concept_training_metadata import resolve_concept_training_metadata


def _write_image(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 100, 100)).save(path)


async def _dataset_with_image(
    datasets_service: DatasetsService,
    tmp_path: Path,
    *,
    filename: str = "sample.png",
    enable_bucket: bool = False,
) -> tuple[object, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / filename)
    dataset = await datasets_service.create_dataset(name="reconcile", image_dir=str(image_dir))
    dataset = await datasets_service.update_dataset(
        dataset.id,
        name=None,
        image_dir=None,
        caption_dir=None,
        description=None,
        target_resolution=1024,
        update_target_resolution=True,
        enable_bucket=enable_bucket,
        update_bucket_settings=True,
    )
    return dataset, image_dir


@pytest.mark.asyncio
async def test_reconcile_removes_orphan_crop_and_prepared(
    datasets_service: DatasetsService,
    tmp_path: Path,
) -> None:
    dataset, image_dir = await _dataset_with_image(datasets_service, tmp_path)
    await datasets_service.save_crop(dataset, "sample.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "sample.png")

    (image_dir / "sample.png").unlink()
    prepared = prepared_dir_path(image_dir, 1024) / "sample.jpg"
    assert prepared.is_file()

    result = await datasets_service.reconcile_dataset(dataset)

    assert result.removed_orphans == ["sample.png"]
    assert not prepared.is_file()
    crops = await datasets_service._crop_repo.list_by_dataset(dataset.id)  # type: ignore[arg-type]
    assert crops == []


@pytest.mark.asyncio
async def test_reconcile_removes_orphan_prepared_without_source(
    datasets_service: DatasetsService,
    tmp_path: Path,
) -> None:
    dataset, image_dir = await _dataset_with_image(datasets_service, tmp_path, filename="keep.png")
    await datasets_service.save_crop(dataset, "keep.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "keep.png")

    prepared_dir = prepared_dir_path(image_dir, 1024)
    orphan_prepared = prepared_dir / "ghost.jpg"
    Image.new("RGB", (1024, 1024), (0, 0, 0)).save(orphan_prepared)

    result = await datasets_service.reconcile_dataset(dataset)

    assert "ghost.jpg" in result.removed_prepared_orphans
    assert not orphan_prepared.is_file()
    assert (prepared_dir / "keep.jpg").is_file()


@pytest.mark.asyncio
async def test_reconcile_renames_file_by_content_hash(
    datasets_service: DatasetsService,
    tmp_path: Path,
) -> None:
    dataset, image_dir = await _dataset_with_image(datasets_service, tmp_path, filename="old.png")
    await datasets_service.save_crop(dataset, "old.png", 0.4, 0.6)
    await datasets_service.bake_image(dataset, "old.png")
    await datasets_service.reconcile_dataset(dataset)

    (image_dir / "old.png").rename(image_dir / "new.png")
    old_prepared = prepared_dir_path(image_dir, 1024) / "old.jpg"
    assert old_prepared.is_file()

    result = await datasets_service.reconcile_dataset(dataset)

    assert result.renamed == [("old.png", "new.png")]
    assert not old_prepared.is_file()
    crop = await datasets_service._crop_repo.get_by_dataset_and_filename(dataset.id, "new.png")  # type: ignore[arg-type]
    assert crop is not None
    assert crop.baked_at is None
    assert crop.crop_center_x == pytest.approx(0.4)
    assert await datasets_service._crop_repo.get_by_dataset_and_filename(dataset.id, "old.png") is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_list_items_with_states_reconciles_orphan_crop(
    datasets_service: DatasetsService,
    tmp_path: Path,
) -> None:
    dataset, image_dir = await _dataset_with_image(datasets_service, tmp_path)
    await datasets_service.save_crop(dataset, "sample.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "sample.png")
    (image_dir / "sample.png").unlink()

    items = await datasets_service.list_items_with_states(dataset)

    assert items == []
    crops = await datasets_service._crop_repo.list_by_dataset(dataset.id)  # type: ignore[arg-type]
    assert crops == []


def test_get_image_state_requires_bucket_metadata_when_baked() -> None:
    bucket_config = BucketPreprocessConfig(
        enable_bucket=True,
        resolution=1024,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_reso_steps=64,
        bucket_no_upscale=True,
    )
    record = StoredCropRecord(
        crop_center_x=0.5,
        crop_center_y=0.5,
        source_mtime=1.0,
        baked_at=datetime.now(timezone.utc),
        bucket_width=None,
        bucket_height=None,
    )
    assert not has_complete_bucket_metadata(record)


@pytest.mark.asyncio
async def test_bake_all_rebakes_ready_image_with_missing_bucket_metadata(
    datasets_service: DatasetsService,
    tmp_path: Path,
) -> None:
    dataset, image_dir = await _dataset_with_image(datasets_service, tmp_path, enable_bucket=True)
    await datasets_service.save_crop(dataset, "sample.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "sample.png")

    crop = await datasets_service._crop_repo.get_by_dataset_and_filename(dataset.id, "sample.png")  # type: ignore[arg-type]
    assert crop is not None
    crop.bucket_width = None
    crop.bucket_height = None
    crop.scale_to_width = None
    crop.scale_to_height = None
    datasets_service._crop_repo._session.add(crop)
    await datasets_service._crop_repo._session.flush()

    baked = await datasets_service.bake_all(dataset)

    assert baked == 1
    crop = await datasets_service._crop_repo.get_by_dataset_and_filename(dataset.id, "sample.png")  # type: ignore[arg-type]
    assert crop is not None
    assert crop.bucket_width is not None
    assert crop.bucket_height is not None


@pytest.mark.asyncio
async def test_resolve_concept_training_metadata_ignores_orphan_crops(
    session,
    datasets_service: DatasetsService,
    tmp_path: Path,
) -> None:
    dataset, image_dir = await _dataset_with_image(datasets_service, tmp_path)
    await datasets_service.save_crop(dataset, "sample.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "sample.png")

    orphan = DatasetImageCrop(
        dataset_id=dataset.id,  # type: ignore[arg-type]
        filename="missing.webp",
        crop_center_x=0.5,
        crop_center_y=0.5,
        source_mtime=1.0,
        baked_at=datetime.now(timezone.utc),
    )
    await datasets_service._crop_repo.add(orphan)

    metadata = await resolve_concept_training_metadata(
        [dataset.id],  # type: ignore[arg-type]
        DatasetRepository(session),
        DatasetImageCropRepository(session),
    )

    assert set(metadata[dataset.id].by_filename) == {"sample.png"}  # type: ignore[index]
