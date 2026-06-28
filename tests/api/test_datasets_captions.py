import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import _get_datasets_service, _get_jobs_service
from src.api.main import app
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.session import register_all_tables
from src.services.datasets.service import DatasetsService
from src.services.jobs.service import JobsService


@pytest.mark.asyncio
async def test_dataset_caption_api(tmp_path) -> None:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), color="red").save(image_dir / "cat.png")

    async with factory() as db_session:
        datasets_service = DatasetsService(DatasetRepository(db_session))
        dataset = await datasets_service.create_dataset(name="cats", image_dir=str(image_dir))
        await db_session.commit()

        async def _override_datasets_service():
            yield datasets_service

        async def _override_jobs_service():
            yield JobsService(
                JobRepository(db_session),
                QueueRepository(db_session),
                JobConfigRepository(db_session),
            )

        app.dependency_overrides[_get_datasets_service] = _override_datasets_service
        app.dependency_overrides[_get_jobs_service] = _override_jobs_service
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                items = await client.get(f"/datasets/{dataset.id}/items")
                assert items.status_code == 200
                assert items.json()["items"][0]["filename"] == "cat.png"
                assert items.json()["items"][0]["tags"] == []

                update = await client.put(
                    f"/datasets/{dataset.id}/captions/cat.png",
                    json={"tags": ["1girl", "cat"]},
                )
                assert update.status_code == 200
                assert update.json()["tags"] == ["1girl", "cat"]

                stats = await client.get(f"/datasets/{dataset.id}/tags/stats")
                assert stats.status_code == 200
                assert stats.json()["tags"] == [{"tag": "1girl", "count": 1}, {"tag": "cat", "count": 1}]

                bulk_remove = await client.post(
                    f"/datasets/{dataset.id}/tags/bulk-remove",
                    json={"tag": "1girl"},
                )
                assert bulk_remove.status_code == 200
                assert bulk_remove.json()["updated_count"] == 1

                caption = await client.get(f"/datasets/{dataset.id}/captions/cat.png")
                assert caption.json()["tags"] == ["cat"]

                image = await client.get(f"/datasets/{dataset.id}/images/cat.png?w=32")
                assert image.status_code == 200
                assert image.headers["content-type"].startswith("image/")

                autotag = await client.post(
                    f"/datasets/{dataset.id}/autotag",
                    json={"mode": "if_empty", "enqueue": False},
                )
                assert autotag.status_code == 201
                assert "job_id" in autotag.json()
        finally:
            app.dependency_overrides.clear()

    await engine.dispose()
