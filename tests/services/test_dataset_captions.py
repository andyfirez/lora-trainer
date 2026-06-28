from pathlib import Path

import pytest
from PIL import Image

from src.services.datasets.captions import (
    collect_tag_stats,
    merge_tags,
    parse_tags,
    safe_filename,
    serialize_tags,
    write_tags,
)
from src.services.datasets.exceptions import DatasetImageNotFoundError, InvalidDatasetFilenameError
from src.services.datasets.service import DatasetsService


def test_parse_tags_deduplicates_and_trims() -> None:
    assert parse_tags("1girl, solo, 1girl , solo") == ["1girl", "solo"]


def test_serialize_tags() -> None:
    assert serialize_tags(["1girl", "solo"]) == "1girl, solo"


def test_safe_filename_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        safe_filename("../secret.png")
    with pytest.raises(ValueError):
        safe_filename("")


def test_merge_tags_modes() -> None:
    existing = ["a", "b"]
    new = ["b", "c"]
    assert merge_tags(existing, new, "overwrite") == ["b", "c"]
    assert merge_tags(existing, new, "append") == ["a", "b", "c"]
    assert merge_tags(existing, new, "if_empty") == ["a", "b"]
    assert merge_tags([], new, "if_empty") == ["b", "c"]


def test_collect_tag_stats_sorted_by_count(tmp_path: Path) -> None:
    image_dir = tmp_path
    Image.new("RGB", (8, 8), color="red").save(image_dir / "a.png")
    Image.new("RGB", (8, 8), color="blue").save(image_dir / "b.png")
    write_tags(image_dir, "a.png", ["solo", "1girl"])
    write_tags(image_dir, "b.png", ["solo"])

    stats = collect_tag_stats(image_dir)
    assert [(item.tag, item.count) for item in stats] == [("solo", 2), ("1girl", 1)]


@pytest.mark.asyncio
async def test_datasets_service_caption_crud(tmp_path: Path, datasets_service: DatasetsService) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), color="green").save(image_dir / "photo.png")

    dataset = await datasets_service.create_dataset(name="demo", image_dir=str(image_dir))
    assert datasets_service.get_tags(dataset, "photo.png") == []

    tags = datasets_service.update_tags(dataset, "photo.png", ["1girl", "solo"])
    assert tags == ["1girl", "solo"]
    assert datasets_service.get_tags(dataset, "photo.png") == ["1girl", "solo"]

    updated = datasets_service.bulk_add_tag(dataset, "portrait")
    assert updated == 1
    assert "portrait" in datasets_service.get_tags(dataset, "photo.png")

    removed = datasets_service.bulk_remove_tag(dataset, "solo")
    assert removed == 1
    assert "solo" not in datasets_service.get_tags(dataset, "photo.png")


@pytest.mark.asyncio
async def test_datasets_service_invalid_filename(
    tmp_path: Path,
    datasets_service: DatasetsService,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), color="green").save(image_dir / "photo.png")
    dataset = await datasets_service.create_dataset(name="demo", image_dir=str(image_dir))

    with pytest.raises(InvalidDatasetFilenameError):
        datasets_service.get_tags(dataset, "../photo.png")

    with pytest.raises(DatasetImageNotFoundError):
        datasets_service.get_tags(dataset, "missing.png")
