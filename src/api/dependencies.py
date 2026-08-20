"""FastAPI dependency injection: session → repositories → services."""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.repositories.sampling_repo import SamplingRepository
from src.db.session import session_factory
from src.db.tables.dataset import Dataset
from src.services.datasets.service import DatasetsService
from src.services.files.service import FilesService
from src.services.loras.service import LoraService
from src.services.sampling.service import SamplingService


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(_get_session)]


def _get_lora_repo(session: SessionDep) -> LoraRepository:
    return LoraRepository(session)


def _get_sampling_repo(session: SessionDep) -> SamplingRepository:
    return SamplingRepository(session)


def _get_dataset_repo(session: SessionDep) -> DatasetRepository:
    return DatasetRepository(session)


LoraRepoDep = Annotated[LoraRepository, Depends(_get_lora_repo)]
SamplingRepoDep = Annotated[SamplingRepository, Depends(_get_sampling_repo)]
DatasetRepoDep = Annotated[DatasetRepository, Depends(_get_dataset_repo)]


def _get_lora_service(lora_repo: LoraRepoDep, dataset_repo: DatasetRepoDep) -> LoraService:
    return LoraService(lora_repo, dataset_repo)


def _get_sampling_service(sampling_repo: SamplingRepoDep) -> SamplingService:
    return SamplingService(sampling_repo)


def _get_crop_repo(session: SessionDep) -> DatasetImageCropRepository:
    return DatasetImageCropRepository(session)


def _get_datasets_service(
    dataset_repo: DatasetRepoDep,
    crop_repo: Annotated[DatasetImageCropRepository, Depends(_get_crop_repo)],
) -> DatasetsService:
    return DatasetsService(dataset_repo, crop_repo)


def _get_files_service() -> FilesService:
    return FilesService()


LoraServiceDep = Annotated[LoraService, Depends(_get_lora_service)]
SamplingServiceDep = Annotated[SamplingService, Depends(_get_sampling_service)]
DatasetsServiceDep = Annotated[DatasetsService, Depends(_get_datasets_service)]
FilesServiceDep = Annotated[FilesService, Depends(_get_files_service)]


async def get_dataset_by_id(dataset_id: int, service: DatasetsServiceDep) -> Dataset:
    return await service.get_dataset(dataset_id)


DatasetDep = Annotated[Dataset, Depends(get_dataset_by_id)]
