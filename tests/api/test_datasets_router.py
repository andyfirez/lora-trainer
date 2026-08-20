"""Router mapping: Dataset ORM -> DatasetResponse."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from src.api.routers.datasets import (
    create_dataset,
    get_dataset,
    import_dataset,
    list_datasets,
    update_dataset,
)
from src.api.schemas.datasets import (
    DatasetCreate,
    DatasetImport,
    DatasetResponse,
    DatasetUpdate,
)
from src.db.tables.dataset import Dataset


def _dataset(*, name: str = "demo", relative_path: str = "images") -> Dataset:
    return Dataset(
        id=1,
        name=name,
        relative_path=relative_path,
        description=None,
        target_resolution=1024,
        preprocess_ready=False,
        enable_bucket=False,
        bucket_reso_steps=64,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_no_upscale=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_list_datasets_maps_orm_to_response(storage_roots) -> None:
    (storage_roots["datasets"] / "images").mkdir()
    service = AsyncMock()
    service.list_datasets.return_value = [_dataset()]

    result = await list_datasets(service)

    assert len(result) == 1
    assert isinstance(result[0], DatasetResponse)
    assert result[0].id == 1
    assert result[0].path_missing is False


@pytest.mark.asyncio
async def test_create_get_import_update_map_orm_to_response(storage_roots) -> None:
    (storage_roots["datasets"] / "images").mkdir()
    dataset = _dataset()
    service = AsyncMock()
    service.create_dataset.return_value = dataset
    service.get_dataset.return_value = dataset
    service.import_dataset.return_value = dataset
    service.update_dataset.return_value = dataset

    created = await create_dataset(DatasetCreate(name="demo", relative_path="images"), service)
    fetched = await get_dataset(1, service)
    imported = await import_dataset(
        DatasetImport(name="demo", source_dir="src", relative_path="images"),
        service,
    )
    updated = await update_dataset(1, DatasetUpdate(description="x"), service)

    for response in (created, fetched, imported, updated):
        assert isinstance(response, DatasetResponse)
        assert response.name == "demo"
        assert response.resolved_path != ""
