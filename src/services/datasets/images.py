"""Dataset image bytes, listing, and duplicate handling."""

from io import BytesIO
from pathlib import Path

from PIL import Image

from src.db.tables.dataset import Dataset
from src.services.datasets.captions import (
    DEFAULT_CAPTION_EXTENSION,
    DatasetItem,
    list_dataset_items,
    list_image_filenames,
    safe_filename,
)
from src.services.datasets.duplicates import DuplicateScanResult, remove_duplicate_files, scan_duplicates
from src.services.datasets.exceptions import (
    DatasetImageNotFoundError,
    DatasetTargetResolutionNotSetError,
    InvalidDatasetFilenameError,
)
from src.services.datasets.paths import dataset_image_dir
from src.services.datasets.preprocess import (
    ImagePreprocessState,
    prepared_dir_path,
    resolve_prepared_path,
)
from src.services.datasets.preprocess_service import DatasetPreprocessService


class DatasetImagesService:
    def __init__(self, preprocess_service: DatasetPreprocessService) -> None:
        self._preprocess = preprocess_service

    def list_images(self, dataset: Dataset) -> list[str]:
        return list_image_filenames(Path(dataset_image_dir(dataset)))

    def list_items(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[DatasetItem]:
        return list_dataset_items(Path(dataset_image_dir(dataset)), caption_extension)

    async def list_items_with_states(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[tuple[DatasetItem, ImagePreprocessState]]:
        await self._preprocess.reconcile_dataset(dataset)
        items = self.list_items(dataset, caption_extension)
        crops = await self._preprocess.list_crops(dataset)
        crop_map = {crop.filename: crop for crop in crops}
        return [
            (
                item,
                self._preprocess.get_image_preprocess_state(
                    dataset,
                    item.filename,
                    crop_map.get(item.filename),
                ),
            )
            for item in items
        ]

    def get_image_bytes(
        self,
        dataset: Dataset,
        filename: str,
        *,
        max_width: int | None = None,
    ) -> tuple[bytes, str]:
        path = self._resolve_image_path(dataset, filename)
        return _read_image_bytes(path, max_width=max_width)

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
        return _read_image_bytes(path, max_width=max_width)

    def scan_duplicates(self, dataset: Dataset) -> DuplicateScanResult:
        return scan_duplicates(Path(dataset_image_dir(dataset)))

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
        from src.services.datasets.captions import image_path

        try:
            safe_filename(filename)
            image_path(Path(dataset_image_dir(dataset)), filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc

        remove_duplicate_files(Path(dataset_image_dir(dataset)), [filename], caption_extension)
        await self._preprocess.purge_image_artifacts(dataset, filename)
        await self._preprocess.update_preprocess_ready_flag(dataset)

    def _resolve_prepared_path(self, dataset: Dataset, filename: str) -> Path:
        if dataset.target_resolution is None:
            raise DatasetTargetResolutionNotSetError(dataset.id)  # type: ignore[arg-type]
        prepared_dir = prepared_dir_path(dataset_image_dir(dataset), dataset.target_resolution)
        path = resolve_prepared_path(prepared_dir, filename)
        if path is None:
            raise DatasetImageNotFoundError(filename)
        return path

    def _resolve_image_path(self, dataset: Dataset, filename: str) -> Path:
        from src.services.datasets.captions import image_path

        try:
            safe_filename(filename)
            return image_path(Path(dataset_image_dir(dataset)), filename)
        except ValueError as exc:
            raise InvalidDatasetFilenameError(filename) from exc
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc


def _read_image_bytes(path: Path, *, max_width: int | None) -> tuple[bytes, str]:
    media_type = (
        "image/jpeg"
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else f"image/{path.suffix.lstrip('.')}"
    )
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
