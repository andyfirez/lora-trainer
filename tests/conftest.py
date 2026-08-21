"""Shared pytest fixtures and helpers for the backend test suite."""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.repositories.sampling_repo import SamplingRepository
from src.db.session import register_all_tables
from src.db.tables.dataset import Dataset
from src.db.tables.lora import Lora
from src.services.datasets.service import DatasetsService
from src.services.loras.service import LoraService
from src.services.sampling.service import SamplingService
from src.settings.app_settings import settings


def write_test_image(
    path: Path,
    size: tuple[int, int] = (800, 600),
    color: tuple[int, int, int] = (100, 100, 100),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture(autouse=True)
def storage_roots(tmp_path):
    datasets = tmp_path / "datasets"
    base_models = tmp_path / "base-models"
    lora = tmp_path / "lora"
    for path in (datasets, base_models, lora):
        path.mkdir()
    (base_models / "test-model").mkdir()
    for name in ("alt-model", "changed"):
        (base_models / name).mkdir()
    settings.storage = settings.storage.model_copy(
        update={
            "datasets_root": str(datasets),
            "base_models_root": str(base_models),
            "lora_root": str(lora),
        }
    )
    return {"datasets": datasets, "base_models": base_models, "lora": lora}


async def _prepare_dataset(
    datasets_service: DatasetsService,
    image_dir,
    name: str = "test-dataset",
    *,
    relative_path: str | None = None,
) -> Dataset:
    rel = relative_path or image_dir.name
    write_test_image(image_dir / "test.png", size=(1024, 1024))
    dataset = await datasets_service.create_dataset(name=name, relative_path=rel)
    dataset = await datasets_service.update_dataset(
        dataset.id,
        target_resolution=1024,
    )
    await datasets_service.save_crop(dataset, "test.png", 0.5, 0.5)
    await datasets_service.bake_image(dataset, "test.png")
    return await datasets_service.get_dataset(dataset.id)  # type: ignore[arg-type]


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
async def datasets_service(session: AsyncSession) -> DatasetsService:
    return DatasetsService(DatasetRepository(session), DatasetImageCropRepository(session))


@pytest_asyncio.fixture
async def lora_service(session: AsyncSession) -> LoraService:
    return LoraService(LoraRepository(session), DatasetRepository(session))


@pytest_asyncio.fixture
async def sampling_service(session: AsyncSession) -> SamplingService:
    return SamplingService(SamplingRepository(session))


@pytest_asyncio.fixture
async def training_dataset(datasets_service: DatasetsService, storage_roots) -> Dataset:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    return await _prepare_dataset(datasets_service, image_dir, relative_path="images")


@pytest_asyncio.fixture
async def minimal_training_yaml(training_dataset: Dataset, storage_roots) -> str:
    return f"""base_model_name: test-model
output_dir: ""
concepts:
  - dataset_id: {training_dataset.id}
"""


@pytest_asyncio.fixture
async def create_training_lora(
    lora_service: LoraService,
    training_dataset: Dataset,
) -> Callable[..., Awaitable[Lora]]:
    async def _create(name: str = "test", config_yaml: str | None = None) -> Lora:
        if config_yaml is None:
            config_yaml = f"""base_model_name: test-model
output_dir: ""
concepts:
  - dataset_id: {training_dataset.id}
"""
        return await lora_service.create_lora(name=name, config_yaml=config_yaml)

    return _create


@pytest.fixture
def sampling_output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "sampling-output"
    path.mkdir()
    return path


@pytest.fixture
def minimal_sampling_yaml(sampling_output_dir: Path) -> str:
    return f"""output_dir: {sampling_output_dir.as_posix()}
parameters:
  prompt:
    mode: fixed
    value: test prompt
"""
