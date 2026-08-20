"""Pure tagging orchestration — no asyncio, no DB, no subprocess."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.services.datasets.captions import merge_tags, read_tags, write_tags
from src.services.datasets.training_cache import invalidate_te_cache_for_image
from src.tagger.config import TaggingConfig
from src.tagger.wd14 import WD14Tagger


@dataclass(frozen=True)
class TaggingProgress:
    current: int
    total: int
    message: str


@dataclass(frozen=True)
class TaggingResult:
    tagged_count: int
    skipped_count: int


def tag_dataset(
    image_dir: Path,
    targets: list[str],
    config: TaggingConfig,
    *,
    target_resolution: int | None = None,
    progress_callback: Callable[[TaggingProgress], None] | None = None,
) -> TaggingResult:
    """Run WD14 tagging over *targets* and merge captions on disk."""
    model_repo = config.resolve_model_repo()
    tagger = WD14Tagger(model_repo)
    total = len(targets)
    tagged_count = 0
    skipped_count = 0

    for index, filename in enumerate(targets, start=1):
        image_path = image_dir / filename
        if not image_path.is_file():
            skipped_count += 1
            message = f"Skipped {filename} (missing)"
            if progress_callback is not None:
                progress_callback(TaggingProgress(current=index, total=total, message=message))
            continue

        predicted = tagger.predict(image_path, threshold=config.threshold, strip_rating=config.strip_rating)
        existing = read_tags(image_dir, filename, config.caption_extension)
        merged = merge_tags(existing, predicted, config.mode.value)
        write_tags(image_dir, filename, merged, config.caption_extension)
        invalidate_te_cache_for_image(image_dir, filename, target_resolution)
        tagged_count += 1
        message = f"Tagged {filename} ({index}/{total})"
        if progress_callback is not None:
            progress_callback(TaggingProgress(current=index, total=total, message=message))

    return TaggingResult(tagged_count=tagged_count, skipped_count=skipped_count)
