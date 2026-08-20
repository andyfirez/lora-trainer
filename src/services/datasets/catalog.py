"""Dataset catalog CRUD and disk↔DB sync."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from src.services.datasets.preprocess_service import DatasetPreprocessService

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset import Dataset
from src.services.datasets.captions import list_image_filenames
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
)
from src.services.datasets.import_dataset import copy_dataset_import
from src.services.datasets.preprocess import validate_target_resolution
from src.services.datasets.relocation import find_relocated_dataset
from src.services.datasets.schemas import DatasetUpdateRequest
from src.services.storage.browse import StorageBrowseService
from src.services.storage.catalog_sync import sync_discovered_items
from src.storage.paths import StorageKind, StoragePaths

_BUCKET_FIELDS = frozenset(
    {
        "enable_bucket",
        "bucket_reso_steps",
        "min_bucket_reso",
        "max_bucket_reso",
        "bucket_no_upscale",
    }
)


def _slug_from_relative_path(relative_path: str) -> str:
    name = Path(relative_path).name
    return name or relative_path.replace("/", "-").replace("\\", "-") or "dataset"


class DatasetCatalogService:
    def __init__(
        self,
        dataset_repo: DatasetRepository,
        crop_repo: DatasetImageCropRepository,
        preprocess_service: DatasetPreprocessService | None = None,
    ) -> None:
        self._repo = dataset_repo
        self._crop_repo = crop_repo
        self._preprocess = preprocess_service

    async def list_datasets(self) -> Sequence[Dataset]:
        StoragePaths.ensure_root(StorageKind.DATASETS)
        await self._sync_discovered_datasets()
        datasets = await self._repo.get_all()
        visible: list[Dataset] = []
        for dataset in datasets:
            if StoragePaths.dataset_dir_exists(dataset.relative_path):
                visible.append(dataset)
        return visible

    async def _sync_discovered_datasets(self) -> None:
        browse = StorageBrowseService()
        discovered = browse.discover_dataset_folders()
        all_datasets = list(await self._repo.get_all())
        existing_paths: set[str] = set()
        for dataset in all_datasets:
            existing_paths.add(dataset.relative_path)
            canonical = StoragePaths.to_relative(StorageKind.DATASETS, dataset.relative_path)
            if canonical is not None:
                existing_paths.add(canonical)

        stale_datasets = [
            dataset
            for dataset in all_datasets
            if not StoragePaths.dataset_dir_exists(dataset.relative_path)
        ]
        crop_filenames_by_dataset_id: dict[int, frozenset[str]] = {}
        for dataset in stale_datasets:
            if dataset.id is None:
                continue
            crops = await self._crop_repo.list_by_dataset(dataset.id)
            if crops:
                crop_filenames_by_dataset_id[dataset.id] = frozenset(
                    crop.filename for crop in crops
                )

        def find_relocated(stale_items: list[Dataset], relative_path: str) -> Dataset | None:
            disk_path = StoragePaths.resolve(StorageKind.DATASETS, relative_path)
            disk_image_filenames = frozenset(list_image_filenames(disk_path))
            return find_relocated_dataset(
                stale_items,
                relative_path,
                disk_image_filenames=disk_image_filenames,
                crop_filenames_by_dataset_id=crop_filenames_by_dataset_id,
            )

        async def make_unique_name(relative_path: str) -> str:
            name = _slug_from_relative_path(relative_path)
            candidate = name
            suffix = 1
            while await self._repo.get_by_name(candidate) is not None:
                suffix += 1
                candidate = f"{name}-{suffix}"
            return candidate

        async def create_entity(relative_path: str, unique_name: str) -> None:
            await self._repo.add(Dataset(name=unique_name, relative_path=relative_path))

        def apply_relocation(relocated: Dataset, relative_path: str) -> None:
            relocated.relative_path = relative_path

        await sync_discovered_items(
            discovered=discovered,
            stale_items=stale_datasets,
            existing_paths=existing_paths,
            get_discovered_path=lambda relative_path: relative_path,
            find_relocated=find_relocated,
            apply_relocation=apply_relocation,
            stage=self._repo.save,
            flush=self._repo.flush,
            make_unique_name=make_unique_name,
            create_entity=create_entity,
        )

    async def get_dataset(self, dataset_id: int) -> Dataset:
        dataset = await self._repo.get_by_id(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        if not StoragePaths.dataset_dir_exists(dataset.relative_path):
            raise DatasetDirectoryNotFoundError(dataset.relative_path)
        return dataset

    async def create_dataset(
        self,
        name: str,
        relative_path: str,
        description: Optional[str] = None,
    ) -> Dataset:
        normalized = StoragePaths.normalize_input_path(StorageKind.DATASETS, relative_path)
        validated = StoragePaths.validate_relative_path(StorageKind.DATASETS, normalized)
        if not StoragePaths.dataset_dir_exists(validated):
            raise DatasetDirectoryNotFoundError(validated)
        existing = await self._repo.get_by_name(name)
        if existing is not None:
            raise DatasetNameConflictError(name)
        dataset = Dataset(name=name, relative_path=validated, description=description)
        return await self._repo.add(dataset)

    async def import_dataset(
        self,
        *,
        name: str,
        source_dir: str,
        relative_path: str,
        description: Optional[str] = None,
    ) -> Dataset:
        normalized = StoragePaths.normalize_input_path(StorageKind.DATASETS, relative_path)
        validated = StoragePaths.validate_relative_path(StorageKind.DATASETS, normalized)
        copy_dataset_import(Path(source_dir), validated)
        return await self.create_dataset(name=name, relative_path=validated, description=description)

    async def update_dataset(self, dataset_id: int, request: DatasetUpdateRequest) -> Dataset:
        dataset = await self.get_dataset(dataset_id)
        if request.was_provided("name") and request.name is not None and request.name != dataset.name:
            existing = await self._repo.get_by_name(request.name)
            if existing is not None:
                raise DatasetNameConflictError(request.name)
            dataset.name = request.name
        if request.was_provided("relative_path") and request.relative_path is not None:
            await self._apply_path_change(dataset, dataset_id, request.relative_path)
        if request.was_provided("description") and request.description is not None:
            dataset.description = request.description
        if request.was_provided("target_resolution"):
            await self._apply_target_resolution(dataset, dataset_id, request.target_resolution)
        if self._has_bucket_fields(request):
            await self._apply_bucket_settings(dataset, request)
        dataset.updated_at = datetime.now(timezone.utc)
        return await self._repo.save_flush_refresh(dataset)

    async def _apply_path_change(self, dataset: Dataset, dataset_id: int, relative_path: str) -> None:
        normalized = StoragePaths.normalize_input_path(StorageKind.DATASETS, relative_path)
        validated = StoragePaths.validate_relative_path(StorageKind.DATASETS, normalized)
        if not StoragePaths.dataset_dir_exists(validated):
            raise DatasetDirectoryNotFoundError(validated)
        if validated == dataset.relative_path:
            return
        dataset.relative_path = validated
        dataset.preprocess_ready = False
        await self._crop_repo.delete_by_dataset(dataset_id)

    async def _apply_target_resolution(
        self,
        dataset: Dataset,
        dataset_id: int,
        target_resolution: int | None,
    ) -> None:
        if target_resolution is not None:
            validate_target_resolution(int(target_resolution))
        normalized = int(target_resolution) if target_resolution is not None else None
        if normalized == dataset.target_resolution:
            return
        dataset.target_resolution = normalized
        dataset.preprocess_ready = False
        await self._crop_repo.delete_by_dataset(dataset_id)

    @staticmethod
    def _has_bucket_fields(request: DatasetUpdateRequest) -> bool:
        return bool(request.provided & _BUCKET_FIELDS)

    async def _apply_bucket_settings(self, dataset: Dataset, request: DatasetUpdateRequest) -> None:
        bucket_changed = False
        if request.enable_bucket is not None and request.enable_bucket != dataset.enable_bucket:
            dataset.enable_bucket = request.enable_bucket
            bucket_changed = True
        if request.bucket_reso_steps is not None and request.bucket_reso_steps != dataset.bucket_reso_steps:
            dataset.bucket_reso_steps = request.bucket_reso_steps
            bucket_changed = True
        if request.min_bucket_reso is not None and request.min_bucket_reso != dataset.min_bucket_reso:
            dataset.min_bucket_reso = request.min_bucket_reso
            bucket_changed = True
        if request.max_bucket_reso is not None and request.max_bucket_reso != dataset.max_bucket_reso:
            dataset.max_bucket_reso = request.max_bucket_reso
            bucket_changed = True
        if request.bucket_no_upscale is not None and request.bucket_no_upscale != dataset.bucket_no_upscale:
            dataset.bucket_no_upscale = request.bucket_no_upscale
            bucket_changed = True
        if not bucket_changed:
            return
        dataset.preprocess_ready = False
        if self._preprocess is not None:
            await self._preprocess.invalidate_prepared_outputs(dataset)

    async def delete_dataset(self, dataset_id: int) -> None:
        dataset = await self.get_dataset(dataset_id)
        await self._repo.delete(dataset)
