"""Caption/tag file utilities for dataset images."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CAPTION_EXTENSION = ".txt"


@dataclass(frozen=True)
class TagStat:
    tag: str
    count: int


@dataclass(frozen=True)
class DatasetItem:
    filename: str
    tags: list[str]
    has_caption: bool


def safe_filename(filename: str) -> str:
    """Reject path traversal in user-supplied filenames."""
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Invalid filename: {filename!r}")
    return filename


def caption_path(image_dir: Path, filename: str, caption_extension: str = DEFAULT_CAPTION_EXTENSION) -> Path:
    safe_filename(filename)
    image_path = image_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(filename)
    return image_path.with_suffix(caption_extension)


def image_path(image_dir: Path, filename: str) -> Path:
    safe_filename(filename)
    path = image_dir / filename
    if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise FileNotFoundError(filename)
    return path


def parse_tags(text: str) -> list[str]:
    seen: set[str] = set()
    tags: list[str] = []
    for part in text.split(","):
        tag = part.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def serialize_tags(tags: list[str]) -> str:
    return ", ".join(tag.strip() for tag in tags if tag.strip())


def read_tags(image_dir: Path, filename: str, caption_extension: str = DEFAULT_CAPTION_EXTENSION) -> list[str]:
    path = caption_path(image_dir, filename, caption_extension)
    if not path.is_file():
        return []
    return parse_tags(path.read_text(encoding="utf-8"))


def write_tags(
    image_dir: Path,
    filename: str,
    tags: list[str],
    caption_extension: str = DEFAULT_CAPTION_EXTENSION,
) -> None:
    path = caption_path(image_dir, filename, caption_extension)
    path.write_text(serialize_tags(tags), encoding="utf-8")


def list_image_filenames(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def list_dataset_items(image_dir: Path, caption_extension: str = DEFAULT_CAPTION_EXTENSION) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for filename in list_image_filenames(image_dir):
        caption_file = image_dir / f"{Path(filename).stem}{caption_extension}"
        has_caption = caption_file.is_file()
        tags = read_tags(image_dir, filename, caption_extension) if has_caption else []
        items.append(DatasetItem(filename=filename, tags=tags, has_caption=has_caption))
    return items


def collect_tag_stats(image_dir: Path, caption_extension: str = DEFAULT_CAPTION_EXTENSION) -> list[TagStat]:
    counter: Counter[str] = Counter()
    for item in list_dataset_items(image_dir, caption_extension):
        counter.update(item.tags)
    return [TagStat(tag=tag, count=count) for tag, count in counter.most_common()]


def merge_tags(existing: list[str], new_tags: list[str], mode: str) -> list[str]:
    if mode == "overwrite":
        return list(new_tags)
    if mode == "append":
        merged = list(existing)
        seen = set(existing)
        for tag in new_tags:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
        return merged
    if mode == "if_empty":
        return list(new_tags) if not existing else list(existing)
    raise ValueError(f"Unknown merge mode: {mode}")
