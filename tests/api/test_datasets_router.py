"""Router mapping: Dataset ORM -> DatasetResponse."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.dependencies import get_dataset_by_id
from src.api.routers.datasets_crud import (
    create_dataset,
    get_dataset,
    import_dataset,
    list_datasets,
    update_dataset,
)
from src.api.routers.datasets_images import get_duplicates
from src.api.routers.datasets_tags import autotag_dataset, get_autotag_status
from src.api.schemas.datasets import (
    AutotagRequest,
    DatasetCreate,
    DatasetImport,
    DatasetResponse,
    DatasetUpdate,
)
from src.db.tables.dataset import Dataset
from src.services.datasets.duplicates import DuplicateScanResult
from src.services.tagging.manager import TaggingStatus, TaggingTaskState


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


class TestDatasetResponseMapping:
    def test_dataset_response_from_orm_object(self, storage_roots) -> None:
        dataset = _dataset()
        (storage_roots["datasets"] / "images").mkdir()

        response = DatasetResponse.model_validate(dataset)
        assert response.relative_path == "images"
        assert response.resolved_path == str((storage_roots["datasets"] / "images").resolve())
        assert response.path_missing is False


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
    fetched = await get_dataset(dataset)
    imported = await import_dataset(
        DatasetImport(name="demo", source_dir="src", relative_path="images"),
        service,
    )
    updated = await update_dataset(1, DatasetUpdate(description="x"), service)

    expected_path = str((storage_roots["datasets"] / "images").resolve())
    service.update_dataset.assert_awaited_once_with(1, description="x")
    for response in (created, fetched, imported, updated):
        assert isinstance(response, DatasetResponse)
        assert response.name == "demo"
        assert response.resolved_path == expected_path
        assert response.path_missing is False


@pytest.mark.asyncio
async def test_get_dataset_by_id_loads_dataset() -> None:
    service = AsyncMock()
    dataset = _dataset()
    service.get_dataset.return_value = dataset

    result = await get_dataset_by_id(1, service)

    assert result is dataset
    service.get_dataset.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_get_duplicates_uses_injected_dataset() -> None:
    service = AsyncMock()
    dataset = _dataset()
    service.scan_duplicates = MagicMock(
        return_value=DuplicateScanResult(duplicate_count=2, duplicate_filenames=("b.png",))
    )

    result = await get_duplicates(dataset, service)

    service.get_dataset.assert_not_called()
    service.scan_duplicates.assert_called_once_with(dataset)
    assert result.duplicate_count == 2


@pytest.mark.asyncio
async def test_autotag_endpoints_delegate_to_service() -> None:
    dataset = _dataset()
    running = TaggingTaskState(status=TaggingStatus.RUNNING, current=1, total=2, message="go")
    idle = TaggingTaskState(status=TaggingStatus.IDLE, current=0, total=0, message="")
    service = MagicMock()
    service.start_autotag.return_value = running
    service.get_autotag_status.return_value = idle

    started = await autotag_dataset(AutotagRequest(threshold=0.5), dataset, service)
    status = await get_autotag_status(dataset, service)

    service.start_autotag.assert_called_once()
    assert started.status == "running"
    assert started.current == 1
    assert status.status == "idle"
