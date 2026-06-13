"""FastAPI dependency injection: session → repositories → services."""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.session import session_factory
from src.services.datasets.service import DatasetsService
from src.services.files.service import FilesService
from src.services.jobs.service import JobsService
from src.services.queues.service import QueuesService


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(_get_session)]


def _get_job_repo(session: SessionDep) -> TrainingJobRepository:
    return TrainingJobRepository(session)


def _get_queue_repo(session: SessionDep) -> QueueRepository:
    return QueueRepository(session)


def _get_dataset_repo(session: SessionDep) -> DatasetRepository:
    return DatasetRepository(session)


JobRepoDep = Annotated[TrainingJobRepository, Depends(_get_job_repo)]
QueueRepoDep = Annotated[QueueRepository, Depends(_get_queue_repo)]
DatasetRepoDep = Annotated[DatasetRepository, Depends(_get_dataset_repo)]


def _get_jobs_service(
    job_repo: JobRepoDep,
    queue_repo: QueueRepoDep,
) -> JobsService:
    return JobsService(job_repo, queue_repo)


def _get_queues_service(
    queue_repo: QueueRepoDep,
    job_repo: JobRepoDep,
) -> QueuesService:
    return QueuesService(queue_repo, job_repo)


def _get_datasets_service(dataset_repo: DatasetRepoDep) -> DatasetsService:
    return DatasetsService(dataset_repo)


def _get_files_service() -> FilesService:
    return FilesService()


JobsServiceDep = Annotated[JobsService, Depends(_get_jobs_service)]
QueuesServiceDep = Annotated[QueuesService, Depends(_get_queues_service)]
DatasetsServiceDep = Annotated[DatasetsService, Depends(_get_datasets_service)]
FilesServiceDep = Annotated[FilesService, Depends(_get_files_service)]
