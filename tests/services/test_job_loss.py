import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import _get_jobs_service
from src.api.main import app
from src.db.migrations import migrate_schema
from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.training_job import TrainingJob
from src.services.jobs.service import JobsService
from src.trainer.metric_logger import MetricLogger


CONFIG_YAML = """
output_dir: output
lora_name: test_lora
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - image_dir: /tmp/images
"""


@pytest.mark.asyncio
async def test_get_job_loss_api_endpoint(tmp_path) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    output_dir = tmp_path / "output"
    config_yaml = CONFIG_YAML.replace("output_dir: output", f"output_dir: {output_dir.as_posix()}")

    async with factory() as db_session:
        job = TrainingJob(name="loss-api-test", config_yaml=config_yaml)
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        loss_log = output_dir / "test_lora" / "loss_log.db"
        logger = MetricLogger(loss_log)
        for step in range(1, 4):
            logger.log({"loss/loss": float(step) / 10})
            logger.commit(step=step)
        logger.finish()

        async def _override_jobs_service():
            yield JobsService(TrainingJobRepository(db_session), QueueRepository(db_session))

        app.dependency_overrides[_get_jobs_service] = _override_jobs_service
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(f"/jobs/{job.id}/loss?key=loss/loss")
                assert response.status_code == 200
                data = response.json()
                assert "loss/loss" in data["keys"]
                assert len(data["points"]) == 3
                assert data["points"][0]["step"] == 1
                assert data["points"][-1]["value"] == pytest.approx(0.3)

                since = await client.get(f"/jobs/{job.id}/loss?key=loss/loss&since_step=2")
                assert since.status_code == 200
                since_points = since.json()["points"]
                assert [p["step"] for p in since_points] == [3]
        finally:
            app.dependency_overrides.clear()

    await engine.dispose()


@pytest.mark.asyncio
async def test_get_job_loss_empty_when_no_file() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as db_session:
        job = TrainingJob(name="no-loss", config_yaml=CONFIG_YAML)
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        async def _override_jobs_service():
            yield JobsService(TrainingJobRepository(db_session), QueueRepository(db_session))

        app.dependency_overrides[_get_jobs_service] = _override_jobs_service
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(f"/jobs/{job.id}/loss")
                assert response.status_code == 200
                assert response.json()["keys"] == []
                assert response.json()["points"] == []
        finally:
            app.dependency_overrides.clear()

    await engine.dispose()
