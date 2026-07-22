"""API integration tests for /trainings and /sampling-configs routers."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.api.dependencies import _get_config_service, _get_jobs_service
from src.api.main import app
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.jobs.service import JobsService


@pytest.fixture
async def api_client(minimal_training_yaml: str):
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as db_session:
        config_service = JobConfigService(JobConfigRepository(db_session), DatasetRepository(db_session))
        jobs_service = JobsService(
            JobRepository(db_session),
            QueueRepository(db_session),
            JobConfigRepository(db_session),
            DatasetRepository(db_session),
        )

        async def _override_config_service():
            yield config_service

        async def _override_jobs_service():
            yield jobs_service

        app.dependency_overrides[_get_config_service] = _override_config_service
        app.dependency_overrides[_get_jobs_service] = _override_jobs_service
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, config_service, minimal_training_yaml
        finally:
            app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_trainings_crud(api_client) -> None:
    client, _, minimal_training_yaml = api_client

    create = await client.post(
        "/trainings/",
        json={"name": "train-a", "config_yaml": minimal_training_yaml, "description": "demo"},
    )
    assert create.status_code == 201
    training_id = create.json()["id"]
    assert create.json()["config_type"] == ConfigType.TRAINING.value

    listing = await client.get("/trainings/")
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    fetched = await client.get(f"/trainings/{training_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "train-a"

    updated = await client.patch(
        f"/trainings/{training_id}",
        json={"name": "train-b"},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "train-b"

    cloned = await client.post(f"/trainings/{training_id}/clone", json={"name": "train-copy"})
    assert cloned.status_code == 201
    assert cloned.json()["name"] == "train-copy"

    deleted = await client.delete(f"/trainings/{training_id}")
    assert deleted.status_code == 204


@pytest.mark.asyncio
async def test_sampling_configs_crud(api_client) -> None:
    client, _, _ = api_client
    sampling_yaml = "sample_prompts:\n  - prompt\n"

    create = await client.post(
        "/sampling-configs/",
        json={"name": "sample-a", "config_yaml": sampling_yaml},
    )
    assert create.status_code == 201
    sampling_id = create.json()["id"]
    assert create.json()["config_type"] == ConfigType.SAMPLING.value

    listing = await client.get("/sampling-configs/")
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    fetched = await client.get(f"/sampling-configs/{sampling_id}")
    assert fetched.status_code == 200

    cloned = await client.post(f"/sampling-configs/{sampling_id}/clone", json={})
    assert cloned.status_code == 201

    deleted = await client.delete(f"/sampling-configs/{sampling_id}")
    assert deleted.status_code == 204


@pytest.mark.asyncio
async def test_type_isolation_returns_404(api_client) -> None:
    client, config_service, minimal_training_yaml = api_client
    sampling_yaml = "sample_prompts:\n  - prompt\n"

    training = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )
    sampling = await config_service.create_config(
        name="sample",
        config_type=ConfigType.SAMPLING,
        config_yaml=sampling_yaml,
    )

    assert (await client.get(f"/trainings/{sampling.id}")).status_code == 404
    assert (await client.get(f"/sampling-configs/{training.id}")).status_code == 404
    assert (await client.patch(f"/trainings/{sampling.id}", json={"name": "x"})).status_code == 404
    assert (await client.delete(f"/sampling-configs/{training.id}")).status_code == 404


@pytest.mark.asyncio
async def test_create_job_from_config(api_client) -> None:
    client, _, minimal_training_yaml = api_client

    create = await client.post(
        "/trainings/",
        json={"name": "train-job", "config_yaml": minimal_training_yaml},
    )
    training_id = create.json()["id"]

    job = await client.post(f"/trainings/{training_id}/jobs", json={"name": "run-1", "enqueue": False})
    assert job.status_code == 201
    assert job.json()["name"] == "run-1"
    assert job.json()["config_id"] == training_id


@pytest.mark.asyncio
async def test_legacy_configs_route_not_found(api_client) -> None:
    client, _, _ = api_client
    response = await client.get("/configs/")
    assert response.status_code == 404
