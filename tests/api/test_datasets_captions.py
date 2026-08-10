import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.api.dependencies import _get_datasets_service
from src.api.main import app
from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.session import register_all_tables
from src.services.datasets.service import DatasetsService


@pytest.mark.asyncio
async def test_dataset_caption_api(storage_roots) -> None:
    register_all_tables()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), color="red").save(image_dir / "cat.png")

    async with factory() as db_session:
        datasets_service = DatasetsService(DatasetRepository(db_session), DatasetImageCropRepository(db_session))
        dataset = await datasets_service.create_dataset(name="cats", relative_path="images")
        await db_session.commit()

        async def _override_datasets_service():
            yield datasets_service

        app.dependency_overrides[_get_datasets_service] = _override_datasets_service
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

                status = await client.get(f"/datasets/{dataset.id}/autotag/status")
                assert status.status_code == 200
                assert status.json()["status"] == "idle"
        finally:
            app.dependency_overrides.clear()

    await engine.dispose()
