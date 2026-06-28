"""Shared pytest fixtures for service tests."""

from collections.abc import Awaitable, Callable

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.db.tables.dataset import Dataset
from src.db.tables.job import Job
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.datasets.service import DatasetsService


def _write_square_image(path, size: int = 1024) -> None:
    from PIL import Image

    Image.new("RGB", (size, size), (100, 100, 100)).save(path)


async def _prepare_dataset(datasets_service: DatasetsService, image_dir, name: str = "test-dataset") -> Dataset:
    _write_square_image(image_dir / "test.png")
    dataset = await datasets_service.create_dataset(name=name, image_dir=str(image_dir))
    dataset = await datasets_service.update_dataset(
        dataset.id,
        name=None,
        image_dir=None,
        caption_dir=None,
        description=None,
        target_resolution=1024,
        update_target_resolution=True,
    )
    await datasets_service.save_crop(dataset, "test.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "test.png")
    return await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]
from src.services.jobs.service import JobsService


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session
    await engine.dispose()


@pytest_asyncio.fixture
async def jobs_service(session: AsyncSession) -> JobsService:
    return JobsService(
        JobRepository(session),
        QueueRepository(session),
        JobConfigRepository(session),
        DatasetRepository(session),
    )


@pytest_asyncio.fixture
async def config_service(session: AsyncSession) -> JobConfigService:
    return JobConfigService(JobConfigRepository(session), DatasetRepository(session))


@pytest_asyncio.fixture
async def datasets_service(session: AsyncSession) -> DatasetsService:
    return DatasetsService(DatasetRepository(session), DatasetImageCropRepository(session))


@pytest_asyncio.fixture
async def training_dataset(datasets_service: DatasetsService, tmp_path) -> Dataset:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    return await _prepare_dataset(datasets_service, image_dir)


@pytest_asyncio.fixture
async def minimal_training_yaml(training_dataset: Dataset) -> str:
    return f"""base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - dataset_id: {training_dataset.id}
"""


@pytest_asyncio.fixture
async def create_training_job(
    jobs_service: JobsService,
    config_service: JobConfigService,
    training_dataset: Dataset,
) -> Callable[..., Awaitable[Job]]:
    async def _create(name: str = "test", config_yaml: str | None = None) -> Job:
        if config_yaml is None:
            config_yaml = f"""base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - dataset_id: {training_dataset.id}
"""
        config = await config_service.create_config(
            name=name,
            config_type=ConfigType.TRAINING,
            config_yaml=config_yaml,
        )
        return await jobs_service.create_from_config(config.id, name=name)

    return _create
