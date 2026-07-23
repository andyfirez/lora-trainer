import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import _get_config_service, _get_jobs_service
from src.api.main import app
from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.db.tables.job_config import ConfigType
from src.services.configs.service import JobConfigService
from src.services.datasets.service import DatasetsService
from src.services.jobs.service import JobsService

SAMPLING_YAML = "sample_prompts:\n  - prompt\n"


@pytest.fixture
async def api_client(tmp_path, storage_roots):
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (32, 32), color="red").save(image_dir / "cat.png")

    async with factory() as db_session:
        datasets_service = DatasetsService(
            DatasetRepository(db_session), DatasetImageCropRepository(db_session)
        )
        dataset = await datasets_service.create_dataset(name="cats", relative_path="images")
        dataset = await datasets_service.update_dataset(
            dataset.id,
            name=None,
            relative_path=None,
            description=None,
            target_resolution=1024,
            update_target_resolution=True,
        )
        await datasets_service.save_crop(dataset, "cat.png", 0.5, 0.5)
        await datasets_service.bake_image(dataset, "cat.png")
        await db_session.commit()

        training_yaml = f"""base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - dataset_id: {dataset.id}
"""

        config_service = JobConfigService(
            JobConfigRepository(db_session), DatasetRepository(db_session)
        )
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
                yield client, config_service, training_yaml
        finally:
            app.dependency_overrides.clear()

    await engine.dispose()


@pytest.mark.asyncio
async def test_trainings_crud_and_type_isolation(api_client) -> None:
    client, config_service, training_yaml = api_client

    create = await client.post(
        "/trainings/",
        json={"name": "train-1", "config_yaml": training_yaml},
    )
    assert create.status_code == 201
    training_id = create.json()["id"]
    assert create.json()["config_type"] == "training"

    sampling = await config_service.create_config(
        name="sample-1",
        config_type=ConfigType.SAMPLING,
        config_yaml=SAMPLING_YAML,
    )

    listed = await client.get("/trainings/")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["name"] == "train-1"

    get_training = await client.get(f"/trainings/{training_id}")
    assert get_training.status_code == 200

    wrong_type = await client.get(f"/trainings/{sampling.id}")
    assert wrong_type.status_code == 404

    patch = await client.patch(
        f"/trainings/{training_id}",
        json={"name": "train-renamed"},
    )
    assert patch.status_code == 200
    assert patch.json()["name"] == "train-renamed"

    clone = await client.post(f"/trainings/{training_id}/clone", json={})
    assert clone.status_code == 201
    assert clone.json()["name"].startswith("train-renamed")

    job = await client.post(
        f"/trainings/{training_id}/jobs",
        json={"name": "job-from-training", "enqueue": False},
    )
    assert job.status_code == 201
    assert job.json()["job_type"] == "training"

    delete = await client.delete(f"/trainings/{training_id}")
    assert delete.status_code == 204

    missing = await client.get(f"/trainings/{training_id}")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_sampling_configs_crud_and_type_isolation(api_client) -> None:
    client, config_service, training_yaml = api_client

    training = await config_service.create_config(
        name="train-1",
        config_type=ConfigType.TRAINING,
        config_yaml=training_yaml,
    )

    create = await client.post(
        "/sampling-configs/",
        json={"name": "sample-1", "config_yaml": SAMPLING_YAML},
    )
    assert create.status_code == 201
    sampling_id = create.json()["id"]
    assert create.json()["config_type"] == "sampling"

    listed = await client.get("/sampling-configs/")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    wrong_type = await client.get(f"/sampling-configs/{training.id}")
    assert wrong_type.status_code == 404

    patch = await client.patch(
        f"/sampling-configs/{sampling_id}",
        json={"description": "updated"},
    )
    assert patch.status_code == 200
    assert patch.json()["description"] == "updated"

    clone = await client.post(f"/sampling-configs/{sampling_id}/clone", json={})
    assert clone.status_code == 201

    delete = await client.delete(f"/sampling-configs/{sampling_id}")
    assert delete.status_code == 204


@pytest.mark.asyncio
async def test_legacy_configs_route_not_found(api_client) -> None:
    client, _, _ = api_client
    response = await client.get("/configs/")
    assert response.status_code == 404
