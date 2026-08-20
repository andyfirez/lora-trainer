"""Tagging service wrapping the in-process autotag manager."""

from pathlib import Path

from src.services.tagging.manager import (
    TaggingTaskManager,
    TaggingTaskState,
    tagging_task_manager,
)
from src.tagger.config import TaggingConfig


class TaggingService:
    def __init__(self, manager: TaggingTaskManager | None = None) -> None:
        self._manager = manager or tagging_task_manager

    def start(
        self,
        dataset_id: int,
        *,
        image_dir: Path,
        config: TaggingConfig,
        target_resolution: int | None,
    ) -> TaggingTaskState:
        return self._manager.start(
            dataset_id,
            image_dir=image_dir,
            config=config,
            target_resolution=target_resolution,
        )

    def get_status(self, dataset_id: int) -> TaggingTaskState | None:
        return self._manager.get_status(dataset_id)
