"""Dataset caption and tag operations."""

from collections.abc import Callable
from pathlib import Path

from src.db.tables.dataset import Dataset
from src.services.datasets.captions import (
    DEFAULT_CAPTION_EXTENSION,
    TagStat,
    parse_tags,
    read_tags,
    safe_filename,
    write_tags,
)
from src.services.datasets.exceptions import (
    DatasetImageNotFoundError,
    DatasetNotFoundError,
    InvalidDatasetFilenameError,
)
from src.services.datasets.paths import dataset_image_dir
from src.services.datasets.training_cache import invalidate_te_cache_for_image
from src.services.tagging.manager import TaggingStatus, TaggingTaskState
from src.services.tagging.service import TaggingService
from src.tagger.config import TaggingConfig


class DatasetTagsService:
    def __init__(self, tagging_service: TaggingService | None = None) -> None:
        self._tagging = tagging_service or TaggingService()

    def start_autotag(
        self,
        dataset: Dataset,
        *,
        mode: str = "if_empty",
        threshold: float = 0.35,
        model: str = "wd-v1-4-convnextv2-tagger-v2",
        caption_extension: str = ".txt",
        strip_rating: bool = True,
        filenames: list[str] | None = None,
    ) -> TaggingTaskState:
        if dataset.id is None:
            raise DatasetNotFoundError(0)
        config = TaggingConfig(
            dataset_id=dataset.id,
            mode=mode,
            threshold=threshold,
            model=model,
            caption_extension=caption_extension,
            strip_rating=strip_rating,
            filenames=filenames or [],
        )
        return self._tagging.start(
            dataset.id,
            image_dir=dataset_image_dir(dataset),
            config=config,
            target_resolution=dataset.target_resolution,
        )

    def get_autotag_status(self, dataset: Dataset) -> TaggingTaskState:
        idle = TaggingTaskState(status=TaggingStatus.IDLE, current=0, total=0, message="")
        if dataset.id is None:
            return idle
        return self._tagging.get_status(dataset.id) or idle

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
            return read_tags(Path(dataset_image_dir(dataset)), filename, caption_extension)
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
            write_tags(Path(dataset_image_dir(dataset)), filename, normalized, caption_extension)
        except FileNotFoundError as exc:
            raise DatasetImageNotFoundError(filename) from exc
        self._invalidate_te_cache(dataset, filename)
        return normalized

    def get_tag_stats(
        self,
        dataset: Dataset,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> list[TagStat]:
        from src.services.datasets.captions import collect_tag_stats

        return collect_tag_stats(Path(dataset_image_dir(dataset)), caption_extension)

    def bulk_add_tag(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None = None,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        def add_tag(tags: list[str], normalized_tag: str) -> bool:
            if normalized_tag in tags:
                return False
            tags.append(normalized_tag)
            return True

        return self._bulk_mutate_tags(dataset, tag, filenames, caption_extension, add_tag)

    def bulk_remove_tag(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None = None,
        caption_extension: str = DEFAULT_CAPTION_EXTENSION,
    ) -> int:
        def remove_tag(tags: list[str], normalized_tag: str) -> bool:
            if normalized_tag not in tags:
                return False
            tags[:] = [item for item in tags if item != normalized_tag]
            return True

        return self._bulk_mutate_tags(dataset, tag, filenames, caption_extension, remove_tag)

    def _bulk_mutate_tags(
        self,
        dataset: Dataset,
        tag: str,
        filenames: list[str] | None,
        caption_extension: str,
        mutate: Callable[[list[str], str], bool],
    ) -> int:
        from src.services.datasets.captions import image_path, list_image_filenames

        normalized_tag = tag.strip()
        if not normalized_tag:
            return 0
        image_dir = Path(dataset_image_dir(dataset))
        targets = filenames if filenames else list_image_filenames(image_dir)
        updated = 0
        for filename in targets:
            try:
                safe_filename(filename)
                image_path(image_dir, filename)
            except (ValueError, FileNotFoundError):
                continue
            tags = read_tags(image_dir, filename, caption_extension)
            if not mutate(tags, normalized_tag):
                continue
            write_tags(image_dir, filename, tags, caption_extension)
            self._invalidate_te_cache(dataset, filename)
            updated += 1
        return updated

    @staticmethod
    def _invalidate_te_cache(dataset: Dataset, filename: str) -> None:
        invalidate_te_cache_for_image(
            dataset_image_dir(dataset),
            filename,
            dataset.target_resolution,
        )
