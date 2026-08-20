"""Dataset CRUD and import routes."""

from fastapi import APIRouter

from src.api.dependencies import DatasetDep, DatasetsServiceDep
from src.api.schemas.datasets import (
    DatasetCreate,
    DatasetImport,
    DatasetResponse,
    DatasetUpdate,
)

router = APIRouter()


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
