"""Business logic for datasets."""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from src.db.repositories.dataset_image_crop_repo import DatasetImageCropRepository
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset import Dataset
from src.db.tables.dataset_image_crop import DatasetImageCrop
from src.services.datasets.captions import (
    DEFAULT_CAPTION_EXTENSION,
    DatasetItem,
    TagStat,
    collect_tag_stats,
    image_path,
    list_dataset_items,
    list_image_filenames,
    parse_tags,
    read_tags,
    safe_filename,
    write_tags,
)
from src.services.datasets.duplicates import DuplicateScanResult, remove_duplicate_files, scan_duplicates
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetImageNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetPreprocessError,
    DatasetTargetResolutionNotSetError,
    InvalidDatasetFilenameError,
)
from src.services.datasets.formats import IMAGE_EXTENSIONS
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
    invalidate_latent_cache_for_prepared,
    is_crop_stale,
    prepared_dir_path,
    recompute_preprocess_ready,
    resolve_prepared_path,
    source_mtime,
    validate_target_resolution,
)
from src.services.datasets.training_cache import invalidate_te_cache_for_image


class DatasetsService:
    def __init__(
        self,
        dataset_repo: DatasetRepository,
        crop_repo: DatasetImageCropRepository,
    ) -> None:
        self._repo = dataset_repo
        self._crop_repo = crop_repo

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
        target_resolution: Optional[int] = None,
        *,
        update_target_resolution: bool = False,
        enable_bucket: Optional[bool] = None,
        bucket_reso_steps: Optional[int] = None,
        min_bucket_reso: Optional[int] = None,
        max_bucket_reso: Optional[int] = None,
        bucket_no_upscale: Optional[bool] = None,
        update_bucket_settings: bool = False,
    ) -> Dataset:
        dataset = await self.get_dataset(dataset_id)
        if name is not None and name != dataset.name:
            existing = await self._repo.get_by_name(name)
            if existing is not None:
                raise DatasetNameConflictError(name)
            dataset.name = name
        if image_dir is not None:
            if not Path(image_dir).is_dir():
                raise DatasetDirectoryNotFoundError(image_dir)
            if image_dir != dataset.image_dir:
                dataset.image_dir = image_dir
                dataset.preprocess_ready = False
                await self._crop_repo.delete_by_dataset(dataset_id)
        if caption_dir is not None:
            dataset.caption_dir = caption_dir
        if description is not None:
            dataset.description = description
        if update_target_resolution:
            if target_resolution is not None:
                validate_target_resolution(target_resolution)
            if target_resolution != dataset.target_resolution:
                dataset.target_resolution = target_resolution
                dataset.preprocess_ready = False
                await self._crop_repo.delete_by_dataset(dataset_id)
        if update_bucket_settings:
            bucket_changed = False
            if enable_bucket is not None and enable_bucket != dataset.enable_bucket:
                dataset.enable_bucket = enable_bucket
                bucket_changed = True
            if bucket_reso_steps is not None and bucket_reso_steps != dataset.bucket_reso_steps:
                dataset.bucket_reso_steps = bucket_reso_steps
                bucket_changed = True
            if min_bucket_reso is not None and min_bucket_reso != dataset.min_bucket_reso:
                dataset.min_bucket_reso = min_bucket_reso
                bucket_changed = True
            if max_bucket_reso is not None and max_bucket_reso != dataset.max_bucket_reso:
                dataset.max_bucket_reso = max_bucket_reso
                bucket_changed = True
            if bucket_no_upscale is not None and bucket_no_upscale != dataset.bucket_no_upscale:
                dataset.bucket_no_upscale = bucket_no_upscale
                bucket_changed = True
            if bucket_changed:
                dataset.preprocess_ready = False
                await self._invalidate_prepared_outputs(dataset)
        dataset.updated_at = datetime.now(timezone.utc)
        self._repo._session.add(dataset)
        await self._repo._session.flush()
        await self._repo._session.refresh(dataset)
        return dataset

    async def delete_dataset(self, dataset_id: int) -> None:
        dataset = await self.get_dataset(dataset_id)
        await self._repo.delete(dataset)

    def list_images(self, dataset: Dataset) -> list[str]:
        return list_image_filenames(Path(dataset.image_dir))

    def list_items(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[DatasetItem]:
        return list_dataset_items(Path(dataset.image_dir), caption_extension)

    def get_tags(
        self,
        dataset: Dataset,
        filename: str,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[str]:
        try:
            safe_filename(filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        try:
            return read_tags(Path(dataset.image_dir), filename, caption_extension)
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc

    def _invalidate_te_cache(self, dataset: Dataset, filename: str) -> None:
        invalidate_te_cache_for_image(
            dataset.image_dir,
            filename,
            dataset.target_resolution,
        )

    def update_tags(
        self,
        dataset: Dataset,
        filename: str,
        tags: list[str],
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[str]:
        try:
            safe_filename(filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        normalized = parse_tags(", ".join(tags))
        try:
            write_tags(Path(dataset.image_dir), filename, normalized, caption_extension)
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc
        self._invalidate_te_cache(dataset, filename)
        return normalized

    def get_tag_stats(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[TagStat]:
        return collect_tag_stats(Path(dataset.image_dir), caption_extension)

    def bulk_add_tag(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None = None,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        normalized_tag = tag.strip()
        if not normalized_tag:
            return 0
        image_dir = Path(dataset.image_dir)
        targets = filenames if filenames else list_image_filenames(image_dir)
        updated = 0
        for filename in targets:
            try:
                safe_filename(filename)
                image_path(image_dir, filename)
            except (ValueError, FileNotFoundError):
                continue
            tags = read_tags(image_dir, filename, caption_extension)
            if normalized_tag in tags:
                continue
            tags.append(normalized_tag)
            write_tags(image_dir, filename, tags, caption_extension)
            self._invalidate_te_cache(dataset, filename)
            updated += 1
        return updated

    def bulk_remove_tag(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None = None,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        normalized_tag = tag.strip()
        if not normalized_tag:
            return 0
        image_dir = Path(dataset.image_dir)
        targets = filenames if filenames else list_image_filenames(image_dir)
        updated = 0
        for filename in targets:
            try:
                safe_filename(filename)
                image_path(image_dir, filename)
            except (ValueError, FileNotFoundError):
                continue
            tags = read_tags(image_dir, filename, caption_extension)
            if normalized_tag not in tags:
                continue
            write_tags(
                image_dir,
                filename,
                [t for t in tags if t != normalized_tag],
                caption_extension,
            )
            self._invalidate_te_cache(dataset, filename)
            updated += 1
        return updated

    def get_image_bytes(
        self,
        dataset: Dataset,
        filename: str,
        *,
        max_width: int | None = None,
    ) -> tuple[bytes, str]:
        try:
            safe_filename(filename)
            path = image_path(Path(dataset.image_dir), filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc

        media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{path.suffix.lstrip('.')}"
        if max_width is None:
            return path.read_bytes(), media_type

        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, max(int(img.height * ratio), 1))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), "image/jpeg"

    def _resolve_prepared_path(self, dataset: Dataset, filename: str) -> Path:
        if dataset.target_resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        prepared_dir = prepared_dir_path(dataset.image_dir, dataset.target_resolution)
        path = resolve_prepared_path(prepared_dir, filename)
        if path is None:
            raise DatasetImageNotFoundError(filename)
        return path

    def get_prepared_image_bytes(
        self,
        dataset: Dataset,
        filename: str,
        *,
        max_width: int | None = None,
    ) -> tuple[bytes, str]:
        try:
            safe_filename(filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        path = self._resolve_prepared_path(dataset, filename)

        media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{path.suffix.lstrip('.')}"
        if max_width is None:
            return path.read_bytes(), media_type

        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, max(int(img.height * ratio), 1))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), "image/jpeg"

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

    async def _invalidate_prepared_outputs(self, dataset: Dataset) -> None:
        if dataset.target_resolution is None:
            return
        prepared_dir = prepared_dir_path(dataset.image_dir, dataset.target_resolution)
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
            self._crop_repo._session.add(crop)
        await self._crop_repo._session.flush()

    async def get_preprocess_status(self, dataset: Dataset) -> PreprocessStatus:
        crop_map = await self._crop_map(dataset.id)  # type: ignore[arg-type]
        return compute_preprocess_status(dataset, crop_map)

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
            self._crop_repo._session.add(existing)
            await self._crop_repo._session.flush()
            crop = existing
        dataset.preprocess_ready = False
        dataset.updated_at = now
        self._repo._session.add(dataset)
        await self._repo._session.flush()
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
        prepared_dir = prepared_dir_path(dataset.image_dir, bucket_config.resolution)
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
        self._crop_repo._session.add(crop)
        invalidate_latent_cache_for_prepared(prepared_path)
        await self._crop_repo._session.flush()
        await self._update_preprocess_ready_flag(dataset)

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
            self._crop_repo._session.add(existing)
            await self._crop_repo._session.flush()
        return existing

    async def bake_all(self, dataset: Dataset, filenames: list[str] | None = None) -> int:
        bucket_config = self._require_bucket_config(dataset)
        all_filenames = filenames if filenames else list_image_filenames(Path(dataset.image_dir))
        image_dir = Path(dataset.image_dir)
        baked = 0
        errors: list[str] = []
        for filename in all_filenames:
            try:
                crop = await self._ensure_crop(dataset, filename)
                state = get_image_state(
                    filename=filename,
                    image_dir=image_dir,
                    bucket_config=bucket_config,
                    crop_record=self._stored_crop_record(crop),
                )
                if state == ImagePreprocessState.READY:
                    continue
                await self.bake_image(dataset, filename)
                baked += 1
            except (DatasetPreprocessError, DatasetImageNotFoundError, InvalidDatasetFilenameError) as exc:
                errors.append(f"{filename}: {exc}")
        if errors and baked == 0:
            raise DatasetPreprocessError("; ".join(errors))
        await self._update_preprocess_ready_flag(dataset)
        if errors:
            raise DatasetPreprocessError(
                f"Baked {baked} image(s); remaining errors: {'; '.join(errors)}"
            )
        return baked

    async def _update_preprocess_ready_flag(self, dataset: Dataset) -> None:
        crop_map = await self._crop_map(dataset.id)  # type: ignore[arg-type]
        dataset.preprocess_ready = recompute_preprocess_ready(dataset, crop_map)
        dataset.updated_at = datetime.now(timezone.utc)
        self._repo._session.add(dataset)
        await self._repo._session.flush()
        await self._repo._session.refresh(dataset)

    def _resolve_image_path(self, dataset: Dataset, filename: str) -> Path:
        try:
            safe_filename(filename)
            return image_path(Path(dataset.image_dir), filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc

    async def list_items_with_states(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[tuple[DatasetItem, ImagePreprocessState]]:
        items = self.list_items(dataset, caption_extension)
        crops = await self._crop_repo.list_by_dataset(dataset.id)  # type: ignore[arg-type]
        crop_map = {crop.filename: crop for crop in crops}
        return [
            (
                item,
                self.get_image_preprocess_state(dataset, item.filename, crop_map.get(item.filename)),
            )
            for item in items
        ]

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

    def scan_duplicates(self, dataset: Dataset) -> DuplicateScanResult:
        return scan_duplicates(Path(dataset.image_dir))

    async def remove_duplicates(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        scan = self.scan_duplicates(dataset)
        if not scan.duplicate_filenames:
            return 0

        removed = 0
        for filename in scan.duplicate_filenames:
            await self.delete_image(dataset, filename, caption_extension)
            removed += 1
        return removed

    async def delete_image(
        self,
        dataset: Dataset,
        filename: str,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> None:
        try:
            safe_filename(filename)
            image_path(Path(dataset.image_dir), filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc

        self._remove_prepared_for_image(dataset, filename)
        self._invalidate_te_cache(dataset, filename)
        remove_duplicate_files(Path(dataset.image_dir), [filename], caption_extension)
        await self._crop_repo.delete_by_dataset_and_filenames(
            dataset.id,  # type: ignore[arg-type]
            [filename],
        )
        await self._update_preprocess_ready_flag(dataset)

    def _remove_prepared_for_image(self, dataset: Dataset, filename: str) -> None:
        if dataset.target_resolution is None:
            return
        prepared_dir = prepared_dir_path(dataset.image_dir, dataset.target_resolution)
        prepared_path = resolve_prepared_path(prepared_dir, filename)
        if prepared_path is None or not prepared_path.is_file():
            return
        invalidate_latent_cache_for_prepared(prepared_path)
        prepared_path.unlink(missing_ok=True)
