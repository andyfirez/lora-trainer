"""Datasets router: CRUD + image listing + caption/tag editing."""

from typing import Sequence

from fastapi import APIRouter, Query
from fastapi.responses import Response

from src.api.dependencies import DatasetsServiceDep, JobsServiceDep
from src.api.schemas.datasets import (
    AutotagRequest,
    AutotagResponse,
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
    DatasetItemResponse,
    DatasetItemsResponse,
    DatasetResponse,
    DatasetUpdate,
    PreprocessStatusResponse,
    TagStatResponse,
    TagStatsResponse,
)

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
    fields_set = body.model_fields_set
    bucket_fields = {
        "enable_bucket",
        "bucket_reso_steps",
        "min_bucket_reso",
        "max_bucket_reso",
        "bucket_no_upscale",
    }
    return await service.update_dataset(  # type: ignore[return-value]
        dataset_id,
        name=body.name,
        image_dir=body.image_dir,
        caption_dir=body.caption_dir,
        description=body.description,
        target_resolution=body.target_resolution,
        update_target_resolution="target_resolution" in fields_set,
        enable_bucket=body.enable_bucket,
        bucket_reso_steps=body.bucket_reso_steps,
        min_bucket_reso=body.min_bucket_reso,
        max_bucket_reso=body.max_bucket_reso,
        bucket_no_upscale=body.bucket_no_upscale,
        update_bucket_settings=bool(fields_set & bucket_fields),
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
    rows = await service.list_items_with_states(dataset, caption_extension)
    return DatasetItemsResponse(
        dataset_id=dataset_id,
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


@router.get("/{dataset_id}/images/{filename}/prepared")
async def get_prepared_image(
    dataset_id: int,
    filename: str,
    service: DatasetsServiceDep,
    w: int | None = Query(default=None, ge=32, le=2048),
) -> Response:
    dataset = await service.get_dataset(dataset_id)
    data, media_type = service.get_prepared_image_bytes(dataset, filename, max_width=w)
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


@router.get("/{dataset_id}/preprocess/status", response_model=PreprocessStatusResponse)
async def get_preprocess_status(dataset_id: int, service: DatasetsServiceDep) -> PreprocessStatusResponse:
    dataset = await service.get_dataset(dataset_id)
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
    dataset_id: int,
    filename: str,
    service: DatasetsServiceDep,
) -> CropMetaResponse:
    dataset = await service.get_dataset(dataset_id)
    meta = await service.get_crop_meta(dataset, filename)
    return _crop_meta_response(meta)


@router.get("/{dataset_id}/images/{filename}/crop-preview")
async def get_crop_preview(
    dataset_id: int,
    filename: str,
    service: DatasetsServiceDep,
) -> Response:
    dataset = await service.get_dataset(dataset_id)
    data = service.get_crop_preview_bytes(dataset, filename)
    return Response(content=data, media_type="image/jpeg")


@router.put("/{dataset_id}/images/{filename}/crop", response_model=CropMetaResponse)
async def save_crop(
    dataset_id: int,
    filename: str,
    body: CropUpdateRequest,
    service: DatasetsServiceDep,
) -> CropMetaResponse:
    dataset = await service.get_dataset(dataset_id)
    meta = await service.save_crop(dataset, filename, body.crop_center_x, body.crop_center_y)
    return _crop_meta_response(meta)


@router.post("/{dataset_id}/preprocess/bake", response_model=BakeResponse)
async def bake_preprocess(
    dataset_id: int,
    body: BakeRequest,
    service: DatasetsServiceDep,
) -> BakeResponse:
    dataset = await service.get_dataset(dataset_id)
    baked_count = await service.bake_all(dataset, body.filenames)
    dataset = await service.get_dataset(dataset_id)
    return BakeResponse(baked_count=baked_count, preprocess_ready=dataset.preprocess_ready)


@router.post("/{dataset_id}/images/{filename}/bake", response_model=CropMetaResponse)
async def bake_single_image(
    dataset_id: int,
    filename: str,
    service: DatasetsServiceDep,
) -> CropMetaResponse:
    dataset = await service.get_dataset(dataset_id)
    await service.bake_image(dataset, filename)
    meta = await service.get_crop_meta(dataset, filename)
    return _crop_meta_response(meta)
