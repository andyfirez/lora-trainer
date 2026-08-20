"""Dataset crop, bake, and preprocess status."""

from datetime import datetime, timezone
from pathlib import Path

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset import Dataset
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.services.datasets.captions import list_image_filenames, safe_filename
from src.services.datasets.exceptions import (
    DatasetImageNotFoundError,
    DatasetPreprocessError,
    DatasetTargetResolutionNotSetError,
    InvalidDatasetFilenameError,
)
from src.services.datasets.formats import IMAGE_EXTENSIONS
from src.services.datasets.hashing import file_sha256
from src.services.datasets.paths import dataset_image_dir
from src.services.datasets.preprocess import (
    BucketPreprocessConfig,
    CropMeta,
    ImagePreprocessState,
    PreprocessStatus,
    StoredCropRecord,
    bake_image_to_prepared,
    build_crop_meta,
    build_fitted_preview_bytes,
    compute_preprocess_status,
    default_crop_center,
    get_image_state,
    has_complete_bucket_metadata,
    invalidate_latent_cache_for_prepared,
    is_crop_stale,
    prepared_dir_path,
    recompute_preprocess_ready,
    resolve_prepared_path,
    source_mtime,
)
from src.services.datasets.reconcile import DatasetReconcileResult, reconcile_dataset_records
from src.services.datasets.training_cache import invalidate_te_cache_for_image


class DatasetPreprocessService:
    def __init__(
        self,
        dataset_repo: DatasetRepository,
        crop_repo: DatasetImageCropRepository,
    ) -> None:
        self._repo = dataset_repo
        self._crop_repo = crop_repo

    async def list_crops(self, dataset: Dataset) -> list[DatasetImageCrop]:
        return await self._crop_repo.list_by_dataset(dataset.id)  # type: ignore[arg-type]

    async def get_preprocess_status(self, dataset: Dataset) -> PreprocessStatus:
        await self.reconcile_dataset(dataset)
        crop_map = await self._crop_map(dataset.id)  # type: ignore[arg-type]
        return compute_preprocess_status(dataset, crop_map)

    async def reconcile_dataset(self, dataset: Dataset) -> DatasetReconcileResult:
        result = await reconcile_dataset_records(
            dataset,
            self._crop_repo,
            purge_artifacts=self.purge_image_artifacts,
        )
        if result.changed:
            await self.update_preprocess_ready_flag(dataset)
            result.preprocess_ready_updated = True
        return result

    async def get_crop_meta(self, dataset: Dataset, filename: str) -> CropMeta:
        path = self._resolve_image_path(dataset, filename)
        bucket_config = self._require_bucket_config(dataset)
        crop = await self._crop_repo.get_by_dataset_and_filename(dataset.id, filename)  # type: ignore[arg-type]
        stored = self._stored_crop_record(crop) if crop else None
        return build_crop_meta(
            image_path=path,
            bucket_config=bucket_config,
            crop_center_x=crop.crop_center_x if crop else None,
            crop_center_y=crop.crop_center_y if crop else None,
            stored=stored,
        )

    def get_crop_preview_bytes(self, dataset: Dataset, filename: str) -> bytes:
        path = self._resolve_image_path(dataset, filename)
        bucket_config = self._require_bucket_config(dataset)
        return build_fitted_preview_bytes(path, bucket_config)

    async def save_crop(
        self,
        dataset: Dataset,
        filename: str,
        crop_center_x: float,
        crop_center_y: float,
    ) -> CropMeta:
        bucket_config = self._require_bucket_config(dataset)
        path = self._resolve_image_path(dataset, filename)
        now = datetime.now(timezone.utc)
        mtime = source_mtime(path)
        existing = await self._crop_repo.get_by_dataset_and_filename(dataset.id, filename)  # type: ignore[arg-type]
        if existing is None:
            crop = DatasetImageCrop(
                dataset_id=dataset.id,  # type: ignore[arg-type]
                filename=filename,
                crop_center_x=crop_center_x,
                crop_center_y=crop_center_y,
                source_mtime=mtime,
                baked_at=None,
            )
            await self._crop_repo.add(crop)
        else:
            existing.crop_center_x = crop_center_x
            existing.crop_center_y = crop_center_y
            existing.source_mtime = mtime
            existing.baked_at = None
            existing.bucket_width = None
            existing.bucket_height = None
            existing.scale_to_width = None
            existing.scale_to_height = None
            existing.crop_x = 0
            existing.crop_y = 0
            existing.updated_at = now
            await self._crop_repo.save_and_flush(existing)
            crop = existing
        dataset.preprocess_ready = False
        dataset.updated_at = now
        await self._repo.save_and_flush(dataset)
        return build_crop_meta(
            image_path=path,
            bucket_config=bucket_config,
            crop_center_x=crop.crop_center_x,
            crop_center_y=crop.crop_center_y,
            stored=self._stored_crop_record(crop),
        )

    async def bake_image(self, dataset: Dataset, filename: str) -> None:
        bucket_config = self._require_bucket_config(dataset)
        crop = await self._crop_repo.get_by_dataset_and_filename(dataset.id, filename)  # type: ignore[arg-type]
        if crop is None:
            raise DatasetPreprocessError(f"No crop defined for {filename}")
        path = self._resolve_image_path(dataset, filename)
        prepared_dir = prepared_dir_path(dataset_image_dir(dataset), bucket_config.resolution)
        stored = self._stored_crop_record(crop)
        prepared_path, assignment = bake_image_to_prepared(
            source_path=path,
            prepared_dir=prepared_dir,
            bucket_config=bucket_config,
            center_x=crop.crop_center_x,
            center_y=crop.crop_center_y,
            stored=stored,
        )
        now = datetime.now(timezone.utc)
        crop.baked_at = now
        crop.updated_at = now
        crop.content_hash = file_sha256(path)
        if assignment is not None:
            crop.bucket_width = assignment.bucket_width
            crop.bucket_height = assignment.bucket_height
            crop.scale_to_width = assignment.scale_to_width
            crop.scale_to_height = assignment.scale_to_height
            crop.crop_x = assignment.crop_x
            crop.crop_y = assignment.crop_y
        else:
            crop.bucket_width = bucket_config.resolution
            crop.bucket_height = bucket_config.resolution
            crop.scale_to_width = None
            crop.scale_to_height = None
            crop.crop_x = 0
            crop.crop_y = 0
        await self._crop_repo.save_and_flush(crop)
        invalidate_latent_cache_for_prepared(prepared_path)
        await self.update_preprocess_ready_flag(dataset)

    async def bake_all(self, dataset: Dataset, filenames: list[str] | None = None) -> int:
        await self.reconcile_dataset(dataset)
        bucket_config = self._require_bucket_config(dataset)
        all_filenames = filenames if filenames else list_image_filenames(Path(dataset_image_dir(dataset)))
        image_dir = Path(dataset_image_dir(dataset))
        baked = 0
        errors: list[str] = []
        for filename in all_filenames:
            try:
                crop = await self._ensure_crop(dataset, filename)
                stored = self._stored_crop_record(crop)
                state = get_image_state(
                    filename=filename,
                    image_dir=image_dir,
                    bucket_config=bucket_config,
                    crop_record=stored,
                )
                if state == ImagePreprocessState.READY and (
                    not bucket_config.enable_bucket or has_complete_bucket_metadata(stored)
                ):
                    continue
                await self.bake_image(dataset, filename)
                baked += 1
            except (DatasetPreprocessError, DatasetImageNotFoundError, InvalidDatasetFilenameError) as exc:
                errors.append(f"{filename}: {exc}")
        if errors and baked == 0:
            raise DatasetPreprocessError("; ".join(errors))
        await self.update_preprocess_ready_flag(dataset)
        if errors:
            raise DatasetPreprocessError(
                f"Baked {baked} image(s); remaining errors: {'; '.join(errors)}"
            )
        return baked

    async def invalidate_prepared_outputs(self, dataset: Dataset) -> None:
        if dataset.target_resolution is None:
            return
        prepared_dir = prepared_dir_path(dataset_image_dir(dataset), dataset.target_resolution)
        if not prepared_dir.is_dir():
            return
        for path in prepared_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                path.unlink(missing_ok=True)
                invalidate_latent_cache_for_prepared(path)
            elif path.name.endswith("_sdxl.npz"):
                path.unlink(missing_ok=True)
        crops = await self._crop_repo.list_by_dataset(dataset.id)  # type: ignore[arg-type]
        now = datetime.now(timezone.utc)
        for crop in crops:
            crop.baked_at = None
            crop.bucket_width = None
            crop.bucket_height = None
            crop.scale_to_width = None
            crop.scale_to_height = None
            crop.crop_x = 0
            crop.crop_y = 0
            crop.updated_at = now
            self._crop_repo.save(crop)
        await self._crop_repo.flush()

    async def purge_image_artifacts(self, dataset: Dataset, filename: str) -> None:
        self._remove_prepared_for_image(dataset, filename)
        invalidate_te_cache_for_image(
            dataset_image_dir(dataset),
            filename,
            dataset.target_resolution,
        )
        await self._crop_repo.delete_by_dataset_and_filenames(
            dataset.id,  # type: ignore[arg-type]
            [filename],
        )

    async def update_preprocess_ready_flag(self, dataset: Dataset) -> None:
        crop_map = await self._crop_map(dataset.id)  # type: ignore[arg-type]
        dataset.preprocess_ready = recompute_preprocess_ready(dataset, crop_map)
        dataset.updated_at = datetime.now(timezone.utc)
        await self._repo.save_flush_refresh(dataset)

    def get_image_preprocess_state(
        self,
        dataset: Dataset,
        filename: str,
        crop: DatasetImageCrop | None,
    ) -> ImagePreprocessState:
        bucket_config = BucketPreprocessConfig.from_dataset(dataset)
        if bucket_config is None:
            return ImagePreprocessState.NO_CROP
        return build_crop_meta(
            image_path=self._resolve_image_path(dataset, filename),
            bucket_config=bucket_config,
            crop_center_x=crop.crop_center_x if crop else None,
            crop_center_y=crop.crop_center_y if crop else None,
            stored=self._stored_crop_record(crop) if crop else None,
        ).state

    async def _crop_map(self, dataset_id: int) -> dict[str, StoredCropRecord]:
        crops = await self._crop_repo.list_by_dataset(dataset_id)
        return {crop.filename: self._stored_crop_record(crop) for crop in crops}

    @staticmethod
    def _stored_crop_record(crop: DatasetImageCrop) -> StoredCropRecord:
        return StoredCropRecord(
            crop_center_x=crop.crop_center_x,
            crop_center_y=crop.crop_center_y,
            source_mtime=crop.source_mtime,
            baked_at=crop.baked_at,
            bucket_width=crop.bucket_width,
            bucket_height=crop.bucket_height,
            scale_to_width=crop.scale_to_width,
            scale_to_height=crop.scale_to_height,
            crop_x=crop.crop_x,
            crop_y=crop.crop_y,
        )

    def _require_bucket_config(self, dataset: Dataset) -> BucketPreprocessConfig:
        bucket_config = BucketPreprocessConfig.from_dataset(dataset)
        if bucket_config is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        return bucket_config

    async def _ensure_crop(self, dataset: Dataset, filename: str) -> DatasetImageCrop:
        path = self._resolve_image_path(dataset, filename)
        mtime = source_mtime(path)
        now = datetime.now(timezone.utc)
        existing = await self._crop_repo.get_by_dataset_and_filename(dataset.id, filename)  # type: ignore[arg-type]
        if existing is None:
            cx, cy = default_crop_center(0, 0)
            crop = DatasetImageCrop(
                dataset_id=dataset.id,  # type: ignore[arg-type]
                filename=filename,
                crop_center_x=cx,
                crop_center_y=cy,
                source_mtime=mtime,
                baked_at=None,
            )
            await self._crop_repo.add(crop)
            return crop
        if is_crop_stale(existing.source_mtime, mtime):
            existing.source_mtime = mtime
            existing.baked_at = None
            existing.updated_at = now
            await self._crop_repo.save_and_flush(existing)
        return existing

    def _resolve_image_path(self, dataset: Dataset, filename: str) -> Path:
        from src.services.datasets.captions import image_path

        try:
            safe_filename(filename)
            return image_path(Path(dataset_image_dir(dataset)), filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc

    def _remove_prepared_for_image(self, dataset: Dataset, filename: str) -> None:
        if dataset.target_resolution is None:
            return
        prepared_dir = prepared_dir_path(dataset_image_dir(dataset), dataset.target_resolution)
        prepared_path = resolve_prepared_path(prepared_dir, filename)
        if prepared_path is None or not prepared_path.is_file():
            return
        invalidate_latent_cache_for_prepared(prepared_path)
        prepared_path.unlink(missing_ok=True)
