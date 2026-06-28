"""Datasets router: CRUD + image listing + caption/tag editing."""

from typing import Sequence

from fastapi import APIRouter, Query
from fastapi.responses import Response

from src.api.dependencies import DatasetsServiceDep, JobsServiceDep
from src.api.schemas.datasets import (
    AutotagRequest,
    AutotagResponse,
    BulkTagRequest,
    BulkTagResponse,
    CaptionResponse,
    CaptionUpdateRequest,
    DatasetCreate,
    DatasetImagesResponse,
    DatasetItemResponse,
    DatasetItemsResponse,
    DatasetResponse,
    DatasetUpdate,
    TagStatResponse,
    TagStatsResponse,
)
router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(service: DatasetsServiceDep) -> Sequence[DatasetResponse]:
    return await service.list_datasets()  # type: ignore[return-value]


@router.post("/", response_model=DatasetResponse, status_code=201)
async def create_dataset(body: DatasetCreate, service: DatasetsServiceDep) -> DatasetResponse:
    dataset = await service.create_dataset(
        name=body.name,
        image_dir=body.image_dir,
        caption_dir=body.caption_dir,
        description=body.description,
    )
    return dataset  # type: ignore[return-value]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, service: DatasetsServiceDep) -> DatasetResponse:
    return await service.get_dataset(dataset_id)  # type: ignore[return-value]


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(dataset_id: int, body: DatasetUpdate, service: DatasetsServiceDep) -> DatasetResponse:
    return await service.update_dataset(  # type: ignore[return-value]
        dataset_id,
        name=body.name,
        image_dir=body.image_dir,
        caption_dir=body.caption_dir,
        description=body.description,
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, service: DatasetsServiceDep) -> None:
    await service.delete_dataset(dataset_id)


@router.get("/{dataset_id}/images", response_model=DatasetImagesResponse)
async def list_images(dataset_id: int, service: DatasetsServiceDep) -> DatasetImagesResponse:
    dataset = await service.get_dataset(dataset_id)
    images = service.list_images(dataset)
    return DatasetImagesResponse(
        dataset_id=dataset_id,
        image_dir=dataset.image_dir,
        images=images,
    )


@router.get("/{dataset_id}/items", response_model=DatasetItemsResponse)
async def list_items(
    dataset_id: int,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> DatasetItemsResponse:
    dataset = await service.get_dataset(dataset_id)
    items = service.list_items(dataset, caption_extension)
    return DatasetItemsResponse(
        dataset_id=dataset_id,
        items=[
            DatasetItemResponse(filename=item.filename, tags=item.tags, has_caption=item.has_caption)
            for item in items
        ],
    )


@router.get("/{dataset_id}/images/{filename}")
async def get_image(
    dataset_id: int,
    filename: str,
    service: DatasetsServiceDep,
    w: int | None = Query(default=None, ge=32, le=2048),
) -> Response:
    dataset = await service.get_dataset(dataset_id)
    data, media_type = service.get_image_bytes(dataset, filename, max_width=w)
    return Response(content=data, media_type=media_type)


@router.get("/{dataset_id}/captions/{filename}", response_model=CaptionResponse)
async def get_caption(
    dataset_id: int,
    filename: str,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> CaptionResponse:
    dataset = await service.get_dataset(dataset_id)
    tags = service.get_tags(dataset, filename, caption_extension)
    return CaptionResponse(filename=filename, tags=tags)


@router.put("/{dataset_id}/captions/{filename}", response_model=CaptionResponse)
async def update_caption(
    dataset_id: int,
    filename: str,
    body: CaptionUpdateRequest,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> CaptionResponse:
    dataset = await service.get_dataset(dataset_id)
    tags = service.update_tags(dataset, filename, body.tags, caption_extension)
    return CaptionResponse(filename=filename, tags=tags)


@router.get("/{dataset_id}/tags/stats", response_model=TagStatsResponse)
async def get_tag_stats(
    dataset_id: int,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> TagStatsResponse:
    dataset = await service.get_dataset(dataset_id)
    stats = service.get_tag_stats(dataset, caption_extension)
    return TagStatsResponse(tags=[TagStatResponse(tag=stat.tag, count=stat.count) for stat in stats])


@router.post("/{dataset_id}/tags/bulk-add", response_model=BulkTagResponse)
async def bulk_add_tag(
    dataset_id: int,
    body: BulkTagRequest,
    service: DatasetsServiceDep,
) -> BulkTagResponse:
    dataset = await service.get_dataset(dataset_id)
    updated = service.bulk_add_tag(
        dataset,
        body.tag,
        body.filenames,
        body.caption_extension,
    )
    return BulkTagResponse(updated_count=updated)


@router.post("/{dataset_id}/tags/bulk-remove", response_model=BulkTagResponse)
async def bulk_remove_tag(
    dataset_id: int,
    body: BulkTagRequest,
    service: DatasetsServiceDep,
) -> BulkTagResponse:
    dataset = await service.get_dataset(dataset_id)
    updated = service.bulk_remove_tag(
        dataset,
        body.tag,
        body.filenames,
        body.caption_extension,
    )
    return BulkTagResponse(updated_count=updated)


@router.post("/{dataset_id}/autotag", response_model=AutotagResponse, status_code=201)
async def autotag_dataset(
    dataset_id: int,
    body: AutotagRequest,
    datasets_service: DatasetsServiceDep,
    jobs_service: JobsServiceDep,
) -> AutotagResponse:
    dataset = await datasets_service.get_dataset(dataset_id)
    job = await jobs_service.create_tagging_job(
        dataset_id=dataset_id,
        dataset_name=dataset.name,
        image_dir=dataset.image_dir,
        mode=body.mode,
        threshold=body.threshold,
        model=body.model,
        caption_extension=body.caption_extension,
        strip_rating=body.strip_rating,
        filenames=body.filenames,
    )
    if body.enqueue and job.id is not None:
        await jobs_service.enqueue_job(job.id)
    return AutotagResponse(job_id=job.id)  # type: ignore[arg-type]
