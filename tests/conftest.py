"""Shared pytest fixtures for service tests."""

from collections.abc import Awaitable, Callable

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.db.tables.job import Job
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
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
    return JobsService(JobRepository(session), QueueRepository(session), JobConfigRepository(session))


@pytest_asyncio.fixture
async def config_service(session: AsyncSession) -> JobConfigService:
    return JobConfigService(JobConfigRepository(session))


@pytest_asyncio.fixture
async def create_training_job(
    jobs_service: JobsService,
    config_service: JobConfigService,
) -> Callable[..., Awaitable[Job]]:
    async def _create(name: str = "test", config_yaml: str = "base_model_name: x") -> Job:
        config = await config_service.create_config(
            name=name,
            config_type=ConfigType.TRAINING,
            config_yaml=config_yaml,
        )
        return await jobs_service.create_from_config(config.id, name=name)

    return _create
