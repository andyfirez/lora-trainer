"""Datasets router: CRUD + image listing."""

from typing import Sequence

from fastapi import APIRouter

from src.api.dependencies import DatasetsServiceDep
from src.api.schemas.datasets import (
    DatasetCreate,
    DatasetImagesResponse,
    DatasetResponse,
    DatasetUpdate,
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
