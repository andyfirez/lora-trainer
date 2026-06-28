"""Repository for DatasetImageCrop table."""

from typing import Optional, Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.base_repo import BaseRepository
from src.db.tables.dataset_image_crop import DatasetImageCrop


class DatasetImageCropRepository(BaseRepository[DatasetImageCrop]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(DatasetImageCrop, session)

    async def get_by_dataset_and_filename(
        self,
        dataset_id: int,
        filename: str,
    ) -> Optional[DatasetImageCrop]:
        result = await self._exec(
            select(DatasetImageCrop)
            .where(DatasetImageCrop.dataset_id == dataset_id)
            .where(DatasetImageCrop.filename == filename)
            .limit(1)
        )
        return result.first()

    async def list_by_dataset(self, dataset_id: int) -> Sequence[DatasetImageCrop]:
        result = await self._exec(
            select(DatasetImageCrop)
            .where(DatasetImageCrop.dataset_id == dataset_id)
            .order_by(DatasetImageCrop.filename)
        )
        return result.all()

    async def delete_by_dataset(self, dataset_id: int) -> None:
        crops = await self.list_by_dataset(dataset_id)
        for crop in crops:
            await self.delete(crop)
