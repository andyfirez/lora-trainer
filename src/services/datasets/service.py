"""Facade over focused dataset sub-services."""

from collections.abc import Sequence
from typing import Optional

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset import Dataset
from src.services.datasets.catalog import DatasetCatalogService
from src.services.datasets.captions import DEFAULT_CAPTION_EXTENSION, DatasetItem, TagStat
from src.services.datasets.duplicates import DuplicateScanResult
from src.services.datasets.images import DatasetImagesService
from src.services.datasets.preprocess import CropMeta, ImagePreprocessState, PreprocessStatus
from src.services.datasets.preprocess_service import DatasetPreprocessService
from src.services.datasets.reconcile import DatasetReconcileResult
from src.services.datasets.schemas import DatasetUpdateRequest
from src.services.datasets.tags import DatasetTagsService
from src.services.tagging.manager import TaggingTaskState
from src.services.tagging.service import TaggingService


class DatasetsService:
    """Backward-compatible entry point delegating to catalog/tags/images/preprocess services."""

    def __init__(
        self,
        dataset_repo: DatasetRepository,
        crop_repo: DatasetImageCropRepository,
        tagging_service: TaggingService | None = None,
    ) -> None:
        self._tags = DatasetTagsService(tagging_service)
        self._preprocess = DatasetPreprocessService(dataset_repo, crop_repo)
        self._catalog = DatasetCatalogService(dataset_repo, crop_repo, self._preprocess)
        self._images = DatasetImagesService(self._preprocess)

    @property
    def _crop_repo(self) -> DatasetImageCropRepository:
        return self._preprocess._crop_repo

    @property
    def _repo(self) -> DatasetRepository:
        return self._catalog._repo

    async def list_datasets(self) -> Sequence[Dataset]:
        return await self._catalog.list_datasets()

    async def get_dataset(self, dataset_id: int) -> Dataset:
        return await self._catalog.get_dataset(dataset_id)

    async def create_dataset(
        self,
        name: str,
        relative_path: str,
        description: Optional[str] = None,
    ) -> Dataset:
        return await self._catalog.create_dataset(name, relative_path, description)

    async def import_dataset(
        self,
        *,
        name: str,
        source_dir: str,
        relative_path: str,
        description: Optional[str] = None,
    ) -> Dataset:
        return await self._catalog.import_dataset(
            name=name,
            source_dir=source_dir,
            relative_path=relative_path,
            description=description,
        )

    async def update_dataset(self, dataset_id: int, **fields: object) -> Dataset:
        request = DatasetUpdateRequest.from_fields(**fields)
        return await self._catalog.update_dataset(dataset_id, request)

    async def delete_dataset(self, dataset_id: int) -> None:
        await self._catalog.delete_dataset(dataset_id)

    def start_autotag(self, dataset: Dataset, **kwargs: object) -> TaggingTaskState:
        return self._tags.start_autotag(dataset, **kwargs)  # type: ignore[arg-type]

    def get_autotag_status(self, dataset: Dataset) -> TaggingTaskState:
        return self._tags.get_autotag_status(dataset)

    def list_images(self, dataset: Dataset) -> list[str]:
        return self._images.list_images(dataset)

    def list_items(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[DatasetItem]:
        return self._images.list_items(dataset, caption_extension)

    def get_tags(
        self,
        dataset: Dataset,
        filename: str,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[str]:
        return self._tags.get_tags(dataset, filename, caption_extension)

    def update_tags(
        self,
        dataset: Dataset,
        filename: str,
        tags: list[str],
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[str]:
        return self._tags.update_tags(dataset, filename, tags, caption_extension)

    def get_tag_stats(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[TagStat]:
        return self._tags.get_tag_stats(dataset, caption_extension)

    def bulk_add_tag(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None = None,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        return self._tags.bulk_add_tag(dataset, tag, filenames, caption_extension)

    def bulk_remove_tag(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None = None,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        return self._tags.bulk_remove_tag(dataset, tag, filenames, caption_extension)

    def get_image_bytes(
        self,
        dataset: Dataset,
        filename: str,
        *,
        max_width: int | None = None,
    ) -> tuple[bytes, str]:
        return self._images.get_image_bytes(dataset, filename, max_width=max_width)

    def get_prepared_image_bytes(
        self,
        dataset: Dataset,
        filename: str,
        *,
        max_width: int | None = None,
    ) -> tuple[bytes, str]:
        return self._images.get_prepared_image_bytes(dataset, filename, max_width=max_width)

    async def get_preprocess_status(self, dataset: Dataset) -> PreprocessStatus:
        return await self._preprocess.get_preprocess_status(dataset)

    async def reconcile_dataset(self, dataset: Dataset) -> DatasetReconcileResult:
        return await self._preprocess.reconcile_dataset(dataset)

    async def get_crop_meta(self, dataset: Dataset, filename: str) -> CropMeta:
        return await self._preprocess.get_crop_meta(dataset, filename)

    def get_crop_preview_bytes(self, dataset: Dataset, filename: str) -> bytes:
        return self._preprocess.get_crop_preview_bytes(dataset, filename)

    async def save_crop(
        self,
        dataset: Dataset,
        filename: str,
        crop_center_x: float,
        crop_center_y: float,
    ) -> CropMeta:
        return await self._preprocess.save_crop(dataset, filename, crop_center_x, crop_center_y)

    async def bake_image(self, dataset: Dataset, filename: str) -> None:
        await self._preprocess.bake_image(dataset, filename)

    async def bake_all(self, dataset: Dataset, filenames: list[str] | None = None) -> int:
        return await self._preprocess.bake_all(dataset, filenames)

    async def list_items_with_states(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[tuple[DatasetItem, ImagePreprocessState]]:
        return await self._images.list_items_with_states(dataset, caption_extension)

    def scan_duplicates(self, dataset: Dataset) -> DuplicateScanResult:
        return self._images.scan_duplicates(dataset)

    async def remove_duplicates(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        return await self._images.remove_duplicates(dataset, caption_extension)

    async def delete_image(
        self,
        dataset: Dataset,
        filename: str,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> None:
        await self._images.delete_image(dataset, filename, caption_extension)


async def reconcile_datasets_for_training(
    dataset_ids: list[int],
    dataset_repo: DatasetRepository,
    crop_repo: DatasetImageCropRepository,
) -> None:
    service = DatasetsService(dataset_repo, crop_repo)
    for dataset_id in dict.fromkeys(dataset_ids):
        dataset = await dataset_repo.get_by_id(dataset_id)
        if dataset is not None:
            await service.reconcile_dataset(dataset)
