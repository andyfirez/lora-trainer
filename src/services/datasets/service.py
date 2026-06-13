"""Business logic for datasets."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset import Dataset
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class DatasetsService:
    def __init__(self, dataset_repo: DatasetRepository) -> None:
        self._repo = dataset_repo

    async def list_datasets(self) -> Sequence[Dataset]:
        return await self._repo.get_all()

    async def get_dataset(self, dataset_id: int) -> Dataset:
        dataset = await self._repo.get_by_id(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        return dataset

    async def create_dataset(
        self,
        name: str,
        image_dir: str,
        caption_dir: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dataset:
        if not Path(image_dir).is_dir():
            raise DatasetDirectoryNotFoundError(image_dir)
        existing = await self._repo.get_by_name(name)
        if existing is not None:
            raise DatasetNameConflictError(name)
        dataset = Dataset(name=name, image_dir=image_dir, caption_dir=caption_dir, description=description)
        return await self._repo.add(dataset)

    async def update_dataset(
        self,
        dataset_id: int,
        name: Optional[str],
        image_dir: Optional[str],
        caption_dir: Optional[str],
        description: Optional[str],
    ) -> Dataset:
        dataset = await self.get_dataset(dataset_id)
        if name is not None:
            dataset.name = name
        if image_dir is not None:
            if not Path(image_dir).is_dir():
                raise DatasetDirectoryNotFoundError(image_dir)
            dataset.image_dir = image_dir
        if caption_dir is not None:
            dataset.caption_dir = caption_dir
        if description is not None:
            dataset.description = description
        dataset.updated_at = datetime.now(timezone.utc)
        self._repo._session.add(dataset)
        await self._repo._session.flush()
        await self._repo._session.refresh(dataset)
        return dataset

    async def delete_dataset(self, dataset_id: int) -> None:
        dataset = await self.get_dataset(dataset_id)
        await self._repo.delete(dataset)

    def list_images(self, dataset: Dataset) -> list[str]:
        """Return sorted list of image filenames in the dataset image_dir."""
        image_dir = Path(dataset.image_dir)
        if not image_dir.is_dir():
            return []
        return sorted(
            str(p.name)
            for p in image_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
