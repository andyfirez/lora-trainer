"""Business logic for datasets."""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.tables.dataset import Dataset
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
    InvalidDatasetFilenameError,
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
