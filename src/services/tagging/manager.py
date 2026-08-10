"""In-process fire-and-forget dataset autotagging — no DB table, no subprocess.

Runs entirely inside the API process: a background asyncio task offloads the
blocking WD14 inference to a worker thread, and progress is tracked in a
process-wide in-memory dict keyed by dataset id. State is lost on restart,
which is acceptable for a "kick off a scan, poll while it runs" operation.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.services.datasets.captions import (
    list_image_filenames,
    merge_tags,
    read_tags,
    write_tags,
)
from src.services.datasets.training_cache import invalidate_te_cache_for_image
from src.services.tagging.exceptions import TaggingAlreadyRunningError
from src.tagger.config import TaggingConfig
from src.tagger.wd14 import WD14Tagger

logger = logging.getLogger(__name__)


class TaggingStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaggingTaskState:
    status: TaggingStatus
    total: int = 0
    current: int = 0
    message: str = ""
    error: str | None = None


def _resolve_targets(image_dir: Path, config: TaggingConfig) -> list[str]:
    if config.filenames:
        return list(config.filenames)
    return list_image_filenames(image_dir)


def _run_tagging_sync(
    image_dir: Path,
    targets: list[str],
    config: TaggingConfig,
    target_resolution: int | None,
    state: TaggingTaskState,
) -> None:
    model_repo = config.resolve_model_repo()
    logger.info("Loading WD14 model %s for dataset autotag", model_repo)
    tagger = WD14Tagger(model_repo)
    total = len(targets)
    for index, filename in enumerate(targets, start=1):
        image_path = image_dir / filename
        if not image_path.is_file():
            state.current = index
            state.message = f"Skipped {filename} (missing)"
            continue
        predicted = tagger.predict(image_path, threshold=config.threshold, strip_rating=config.strip_rating)
        existing = read_tags(image_dir, filename, config.caption_extension)
        merged = merge_tags(existing, predicted, config.mode.value)
        write_tags(image_dir, filename, merged, config.caption_extension)
        invalidate_te_cache_for_image(image_dir, filename, target_resolution)
        state.current = index
        state.message = f"Tagged {filename} ({index}/{total})"


class TaggingTaskManager:
    """Process-wide singleton tracking in-flight autotag runs, keyed by dataset id."""

    def __init__(self) -> None:
        self._states: dict[int, TaggingTaskState] = {}

    def is_running(self, dataset_id: int) -> bool:
        state = self._states.get(dataset_id)
        return state is not None and state.status == TaggingStatus.RUNNING

    def get_status(self, dataset_id: int) -> TaggingTaskState | None:
        return self._states.get(dataset_id)

    def start(
        self,
        dataset_id: int,
        *,
        image_dir: Path,
        config: TaggingConfig,
        target_resolution: int | None,
    ) -> TaggingTaskState:
        if self.is_running(dataset_id):
            raise TaggingAlreadyRunningError(dataset_id)
        targets = _resolve_targets(image_dir, config)
        state = TaggingTaskState(status=TaggingStatus.RUNNING, total=len(targets))
        self._states[dataset_id] = state
        if not targets:
            state.status = TaggingStatus.COMPLETED
            state.message = "No images to tag"
            return state
        asyncio.create_task(self._run(dataset_id, image_dir, targets, config, target_resolution, state))
        return state

    async def _run(
        self,
        dataset_id: int,
        image_dir: Path,
        targets: list[str],
        config: TaggingConfig,
        target_resolution: int | None,
        state: TaggingTaskState,
    ) -> None:
        try:
            await asyncio.to_thread(_run_tagging_sync, image_dir, targets, config, target_resolution, state)
            state.status = TaggingStatus.COMPLETED
            state.message = "Completed"
        except Exception as exc:
            logger.exception("Autotag failed for dataset id=%d", dataset_id)
            state.status = TaggingStatus.FAILED
            state.error = str(exc)


tagging_task_manager = TaggingTaskManager()
