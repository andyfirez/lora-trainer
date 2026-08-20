"""Dataset preprocess, crop, and bake routes."""

from fastapi import APIRouter
from fastapi.responses import Response

from src.api.dependencies import DatasetDep, DatasetsServiceDep
from src.api.schemas.datasets import (
    BakeRequest,
    BakeResponse,
    CropMetaResponse,
    CropUpdateRequest,
    PreprocessStatusResponse,
)
from src.services.datasets.preprocess import CropMeta

router = APIRouter()


def _crop_meta_response(meta: CropMeta) -> CropMetaResponse:
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
