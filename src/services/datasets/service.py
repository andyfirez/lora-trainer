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
from src.services.datasets.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetImageNotFoundError,
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetPreprocessError,
    DatasetTargetResolutionNotSetError,
    InvalidDatasetFilenameError,
)
from src.services.datasets.training_cache import invalidate_te_cache_for_image
from src.services.datasets.preprocess import (
    CropMeta,
    ImagePreprocessState,
    PreprocessStatus,
    bake_image_to_prepared,
    build_crop_meta,
    build_fitted_preview_bytes,
    compute_preprocess_status,
    default_crop_center,
    get_image_state,
    is_crop_stale,
    prepared_dir_path,
    recompute_preprocess_ready,
    source_mtime,
    validate_target_resolution,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
        path = prepared_dir / filename
        if path.is_file():
            return path
        alt_png = prepared_dir / f"{Path(filename).stem}.png"
        if alt_png.is_file():
            return alt_png
        raise DatasetImageNotFoundError(filename)

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

    async def _crop_map(self, dataset_id: int) -> dict[str, tuple[float, float, float, datetime | None]]:
        crops = await self._crop_repo.list_by_dataset(dataset_id)
        return {
            crop.filename: (
                crop.crop_center_x,
                crop.crop_center_y,
                crop.source_mtime,
                crop.baked_at,
            )
            for crop in crops
        }

    async def get_preprocess_status(self, dataset: Dataset) -> PreprocessStatus:
        crop_map = await self._crop_map(dataset.id)  # type: ignore[arg-type]
        return compute_preprocess_status(dataset, crop_map)

    async def get_crop_meta(self, dataset: Dataset, filename: str) -> CropMeta:
        path = self._resolve_image_path(dataset, filename)
        resolution = dataset.target_resolution
        if resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        crop = await self._crop_repo.get_by_dataset_and_filename(dataset.id, filename)  # type: ignore[arg-type]
        return build_crop_meta(
            image_path=path,
            resolution=resolution,
            crop_center_x=crop.crop_center_x if crop else None,
            crop_center_y=crop.crop_center_y if crop else None,
            crop_mtime=crop.source_mtime if crop else None,
            crop_baked_at=crop.baked_at if crop else None,
        )

    def get_crop_preview_bytes(self, dataset: Dataset, filename: str) -> bytes:
        path = self._resolve_image_path(dataset, filename)
        if dataset.target_resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        return build_fitted_preview_bytes(path, dataset.target_resolution)

    async def save_crop(
        self,
        dataset: Dataset,
        filename: str,
        crop_center_x: float,
        crop_center_y: float,
    ) -> CropMeta:
        if dataset.target_resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
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
            resolution=dataset.target_resolution,
            crop_center_x=crop.crop_center_x,
            crop_center_y=crop.crop_center_y,
            crop_mtime=crop.source_mtime,
            crop_baked_at=crop.baked_at,
        )

    async def bake_image(self, dataset: Dataset, filename: str) -> None:
        if dataset.target_resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        crop = await self._crop_repo.get_by_dataset_and_filename(dataset.id, filename)  # type: ignore[arg-type]
        if crop is None:
            raise DatasetPreprocessError(f"No crop defined for {filename}")
        path = self._resolve_image_path(dataset, filename)
        prepared_dir = prepared_dir_path(dataset.image_dir, dataset.target_resolution)
        bake_image_to_prepared(
            source_path=path,
            prepared_dir=prepared_dir,
            resolution=dataset.target_resolution,
            center_x=crop.crop_center_x,
            center_y=crop.crop_center_y,
        )
        now = datetime.now(timezone.utc)
        crop.baked_at = now
        crop.updated_at = now
        self._crop_repo._session.add(crop)
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
        if dataset.target_resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        all_filenames = filenames if filenames else list_image_filenames(Path(dataset.image_dir))
        resolution = dataset.target_resolution
        image_dir = Path(dataset.image_dir)
        baked = 0
        errors: list[str] = []
        for filename in all_filenames:
            try:
                crop = await self._ensure_crop(dataset, filename)
                state = get_image_state(
                    filename=filename,
                    image_dir=image_dir,
                    resolution=resolution,
                    crop_mtime=crop.source_mtime,
                    crop_baked_at=crop.baked_at,
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
        if dataset.target_resolution is None:
            return ImagePreprocessState.NO_CROP
        return build_crop_meta(
            image_path=self._resolve_image_path(dataset, filename),
            resolution=dataset.target_resolution,
            crop_center_x=crop.crop_center_x if crop else None,
            crop_center_y=crop.crop_center_y if crop else None,
            crop_mtime=crop.source_mtime if crop else None,
            crop_baked_at=crop.baked_at if crop else None,
        ).state
