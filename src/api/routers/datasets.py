"""Datasets router: CRUD + image listing + caption/tag editing."""

from fastapi import APIRouter, Query
from fastapi.responses import Response

from src.api.dependencies import DatasetDep, DatasetsServiceDep
from src.api.schemas.datasets import (
    AutotagRequest,
    AutotagStatusResponse,
    BakeRequest,
    BakeResponse,
    BulkTagRequest,
    BulkTagResponse,
    CaptionResponse,
    CaptionUpdateRequest,
    CropMetaResponse,
    CropUpdateRequest,
    DatasetCreate,
    DatasetImagesResponse,
    DatasetImport,
    DatasetItemResponse,
    DatasetItemsResponse,
    DatasetResponse,
    DatasetUpdate,
    DuplicatesResponse,
    PreprocessStatusResponse,
    RemoveDuplicatesResponse,
    TagStatResponse,
    TagStatsResponse,
)
from src.services.tagging.manager import TaggingTaskState
from src.storage.paths import StoragePaths

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _crop_meta_response(meta) -> CropMetaResponse:
    return CropMetaResponse(
        crop_center_x=meta.crop_center_x,
        crop_center_y=meta.crop_center_y,
        fitted_width=meta.fitted_width,
        fitted_height=meta.fitted_height,
        source_width=meta.source_width,
        source_height=meta.source_height,
        state=meta.state.value,
        enable_bucket=meta.enable_bucket,
        bucket_width=meta.bucket_width,
        bucket_height=meta.bucket_height,
        scale_to_width=meta.scale_to_width,
        scale_to_height=meta.scale_to_height,
        crop_x=meta.crop_x,
        crop_y=meta.crop_y,
    )


@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(service: DatasetsServiceDep) -> list[DatasetResponse]:
    datasets = await service.list_datasets()
    return [DatasetResponse.model_validate(dataset) for dataset in datasets]


@router.post("/", response_model=DatasetResponse, status_code=201)
async def create_dataset(body: DatasetCreate, service: DatasetsServiceDep) -> DatasetResponse:
    dataset = await service.create_dataset(
        name=body.name,
        relative_path=body.relative_path,
        description=body.description,
    )
    return DatasetResponse.model_validate(dataset)


@router.post("/import", response_model=DatasetResponse, status_code=201)
async def import_dataset(body: DatasetImport, service: DatasetsServiceDep) -> DatasetResponse:
    dataset = await service.import_dataset(
        name=body.name,
        source_dir=body.source_dir,
        relative_path=body.relative_path,
        description=body.description,
    )
    return DatasetResponse.model_validate(dataset)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset: DatasetDep) -> DatasetResponse:
    return DatasetResponse.model_validate(dataset)


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(dataset_id: int, body: DatasetUpdate, service: DatasetsServiceDep) -> DatasetResponse:
    dataset = await service.update_dataset(dataset_id, **body.model_dump(exclude_unset=True))
    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, service: DatasetsServiceDep) -> None:
    await service.delete_dataset(dataset_id)


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


@router.get("/{dataset_id}/captions/{filename}", response_model=CaptionResponse)
async def get_caption(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> CaptionResponse:
    tags = service.get_tags(dataset, filename, caption_extension)
    return CaptionResponse(filename=filename, tags=tags)


@router.put("/{dataset_id}/captions/{filename}", response_model=CaptionResponse)
async def update_caption(
    dataset: DatasetDep,
    filename: str,
    body: CaptionUpdateRequest,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> CaptionResponse:
    tags = service.update_tags(dataset, filename, body.tags, caption_extension)
    return CaptionResponse(filename=filename, tags=tags)


@router.get("/{dataset_id}/tags/stats", response_model=TagStatsResponse)
async def get_tag_stats(
    dataset: DatasetDep,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> TagStatsResponse:
    stats = service.get_tag_stats(dataset, caption_extension)
    return TagStatsResponse(tags=[TagStatResponse(tag=stat.tag, count=stat.count) for stat in stats])


@router.post("/{dataset_id}/tags/bulk-add", response_model=BulkTagResponse)
async def bulk_add_tag(
    dataset: DatasetDep,
    body: BulkTagRequest,
    service: DatasetsServiceDep,
) -> BulkTagResponse:
    updated = service.bulk_add_tag(
        dataset,
        body.tag,
        body.filenames,
        body.caption_extension,
    )
    return BulkTagResponse(updated_count=updated)


@router.post("/{dataset_id}/tags/bulk-remove", response_model=BulkTagResponse)
async def bulk_remove_tag(
    dataset: DatasetDep,
    body: BulkTagRequest,
    service: DatasetsServiceDep,
) -> BulkTagResponse:
    updated = service.bulk_remove_tag(
        dataset,
        body.tag,
        body.filenames,
        body.caption_extension,
    )
    return BulkTagResponse(updated_count=updated)


def _autotag_status_response(state: TaggingTaskState) -> AutotagStatusResponse:
    return AutotagStatusResponse(
        status=state.status.value,
        current=state.current,
        total=state.total,
        message=state.message,
        error=state.error,
    )


@router.post("/{dataset_id}/autotag", response_model=AutotagStatusResponse, status_code=202)
async def autotag_dataset(
    body: AutotagRequest,
    dataset: DatasetDep,
    service: DatasetsServiceDep,
) -> AutotagStatusResponse:
    state = service.start_autotag(dataset, **body.model_dump())
    return _autotag_status_response(state)


@router.get("/{dataset_id}/autotag/status", response_model=AutotagStatusResponse)
async def get_autotag_status(dataset: DatasetDep, service: DatasetsServiceDep) -> AutotagStatusResponse:
    return _autotag_status_response(service.get_autotag_status(dataset))


@router.get("/{dataset_id}/preprocess/status", response_model=PreprocessStatusResponse)
async def get_preprocess_status(dataset: DatasetDep, service: DatasetsServiceDep) -> PreprocessStatusResponse:
    status = await service.get_preprocess_status(dataset)
    return PreprocessStatusResponse(
        target_resolution=status.target_resolution,
        preprocess_ready=status.preprocess_ready,
        total=status.total,
        no_crop=status.no_crop,
        stale=status.stale,
        cropped=status.cropped,
        ready=status.ready,
    )


@router.get("/{dataset_id}/images/{filename}/crop-meta", response_model=CropMetaResponse)
async def get_crop_meta(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
) -> CropMetaResponse:
    meta = await service.get_crop_meta(dataset, filename)
    return _crop_meta_response(meta)


@router.get("/{dataset_id}/images/{filename}/crop-preview")
async def get_crop_preview(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
) -> Response:
    data = service.get_crop_preview_bytes(dataset, filename)
    return Response(content=data, media_type="image/jpeg")


@router.put("/{dataset_id}/images/{filename}/crop", response_model=CropMetaResponse)
async def save_crop(
    dataset: DatasetDep,
    filename: str,
    body: CropUpdateRequest,
    service: DatasetsServiceDep,
) -> CropMetaResponse:
    meta = await service.save_crop(dataset, filename, body.crop_center_x, body.crop_center_y)
    return _crop_meta_response(meta)


@router.post("/{dataset_id}/preprocess/bake", response_model=BakeResponse)
async def bake_preprocess(
    dataset_id: int,
    dataset: DatasetDep,
    body: BakeRequest,
    service: DatasetsServiceDep,
) -> BakeResponse:
    baked_count = await service.bake_all(dataset, body.filenames)
    dataset = await service.get_dataset(dataset_id)
    return BakeResponse(baked_count=baked_count, preprocess_ready=dataset.preprocess_ready)


@router.post("/{dataset_id}/images/{filename}/bake", response_model=CropMetaResponse)
async def bake_single_image(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
) -> CropMetaResponse:
    await service.bake_image(dataset, filename)
    meta = await service.get_crop_meta(dataset, filename)
    return _crop_meta_response(meta)
