"""Dataset image listing, bytes, and duplicate routes."""

from fastapi import APIRouter, Query
from fastapi.responses import Response

from src.api.dependencies import DatasetDep, DatasetsServiceDep
from src.api.schemas.datasets import (
    DatasetImagesResponse,
    DatasetItemResponse,
    DatasetItemsResponse,
    DuplicatesResponse,
    RemoveDuplicatesResponse,
)
from src.storage.paths import StoragePaths

router = APIRouter()


@router.get("/{dataset_id}/duplicates", response_model=DuplicatesResponse)
async def get_duplicates(dataset: DatasetDep, service: DatasetsServiceDep) -> DuplicatesResponse:
    result = service.scan_duplicates(dataset)
    return DuplicatesResponse(duplicate_count=result.duplicate_count)


@router.post("/{dataset_id}/duplicates/remove", response_model=RemoveDuplicatesResponse)
async def remove_duplicates(
    dataset: DatasetDep,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> RemoveDuplicatesResponse:
    removed_count = await service.remove_duplicates(dataset, caption_extension)
    return RemoveDuplicatesResponse(removed_count=removed_count)


@router.get("/{dataset_id}/images", response_model=DatasetImagesResponse)
async def list_images(dataset: DatasetDep, service: DatasetsServiceDep) -> DatasetImagesResponse:
    images = service.list_images(dataset)
    return DatasetImagesResponse(
        dataset_id=dataset.id,
        relative_path=dataset.relative_path,
        resolved_path=str(StoragePaths.resolve_dataset_path(dataset.relative_path)),
        images=images,
    )


@router.get("/{dataset_id}/items", response_model=DatasetItemsResponse)
async def list_items(
    dataset: DatasetDep,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> DatasetItemsResponse:
    rows = await service.list_items_with_states(dataset, caption_extension)
    return DatasetItemsResponse(
        dataset_id=dataset.id,
        items=[
            DatasetItemResponse(
                filename=item.filename,
                tags=item.tags,
                has_caption=item.has_caption,
                preprocess_state=state.value,
            )
            for item, state in rows
        ],
    )


@router.delete("/{dataset_id}/images/{filename}", status_code=204)
async def delete_image(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> None:
    await service.delete_image(dataset, filename, caption_extension)


@router.get("/{dataset_id}/images/{filename}")
async def get_image(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
    w: int | None = Query(default=None, ge=32, le=2048),
) -> Response:
    data, media_type = service.get_image_bytes(dataset, filename, max_width=w)
    return Response(content=data, media_type=media_type)


@router.get("/{dataset_id}/images/{filename}/prepared")
async def get_prepared_image(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
    w: int | None = Query(default=None, ge=32, le=2048),
) -> Response:
    data, media_type = service.get_prepared_image_bytes(dataset, filename, max_width=w)
    return Response(content=data, media_type=media_type)
