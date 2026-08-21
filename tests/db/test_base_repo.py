"""BaseRepository generic query helpers."""

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.tables.dataset import Dataset
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.db.tables.lora import Lora


def _crop(dataset_id: int, filename: str) -> DatasetImageCrop:
    return DatasetImageCrop(
        dataset_id=dataset_id,
        filename=filename,
        crop_center_x=0.5,
        crop_center_y=0.5,
        source_mtime=1.0,
    )


@pytest.mark.asyncio
async def test_list_ordered_sorts_by_created_at_desc(session: AsyncSession) -> None:
    repo = LoraRepository(session)
    await repo.add(Lora(name="older", config_yaml="x: 1"))
    await repo.add(Lora(name="newer", config_yaml="x: 1"))

    rows = await repo.list_ordered(Lora.created_at.desc())

    assert [row.name for row in rows] == ["newer", "older"]


@pytest.mark.asyncio
async def test_get_by_field_returns_match_or_none(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    await repo.add(Dataset(name="demo", relative_path="images"))

    found = await repo.get_by_field("name", "demo")
    missing = await repo.get_by_field("name", "missing")

    assert found is not None
    assert found.relative_path == "images"
    assert missing is None
    assert await repo.get_by_name("demo") is found


@pytest.mark.asyncio
async def test_delete_where_requires_filters(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    with pytest.raises(ValueError, match="at least one filter"):
        await repo.delete_where()


@pytest.mark.asyncio
async def test_delete_where_bulk_deletes_crops(session: AsyncSession) -> None:
    datasets = DatasetRepository(session)
    crops = DatasetImageCropRepository(session)
    dataset = await datasets.add(Dataset(name="demo", relative_path="images"))
    assert dataset.id is not None
    await crops.add(_crop(dataset.id, "a.png"))
    await crops.add(_crop(dataset.id, "b.png"))

    await crops.delete_by_dataset(dataset.id)

    assert list(await crops.list_by_dataset(dataset.id)) == []


@pytest.mark.asyncio
async def test_delete_where_filters_collection(session: AsyncSession) -> None:
    datasets = DatasetRepository(session)
    crops = DatasetImageCropRepository(session)
    dataset = await datasets.add(Dataset(name="demo", relative_path="images"))
    assert dataset.id is not None
    await crops.add(_crop(dataset.id, "keep.png"))
    await crops.add(_crop(dataset.id, "drop.png"))

    await crops.delete_by_dataset_and_filenames(dataset.id, ["drop.png"])

    remaining = [crop.filename for crop in await crops.list_by_dataset(dataset.id)]
    assert remaining == ["keep.png"]


@pytest.mark.asyncio
async def test_lora_repo_round_trips_fields(session: AsyncSession, tmp_path) -> None:
    lora = Lora(name="test-lora", base_model_name="test-model", config_yaml="base_model_name: x")
    session.add(lora)
    await session.commit()
    await session.refresh(lora)

    repo = LoraRepository(session)
    lora.log_path = str(tmp_path / "lora.log")
    session.add(lora)
    await session.commit()

    fetched = await repo.get_by_name("test-lora")
    assert fetched is not None
    assert fetched.log_path == str(tmp_path / "lora.log")
