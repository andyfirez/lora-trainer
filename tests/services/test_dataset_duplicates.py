"""Tests for duplicate image detection and removal."""

from pathlib import Path

import pytest
from conftest import write_test_image
from src.services.datasets.duplicates import remove_duplicate_files, scan_duplicates
from src.services.datasets.service import DatasetsService


def test_scan_duplicates_finds_exact_byte_matches(tmp_path: Path) -> None:
    write_test_image(tmp_path / "a.png", size=(64, 64), color=(100, 120, 140))
    write_test_image(tmp_path / "b.png", size=(64, 64), color=(10, 20, 30))
    (tmp_path / "copy_of_a.png").write_bytes((tmp_path / "a.png").read_bytes())

    result = scan_duplicates(tmp_path)

    assert result.duplicate_count == 1
    assert result.duplicate_filenames == ("copy_of_a.png",)


def test_remove_duplicate_files_deletes_image_and_caption(tmp_path: Path) -> None:
    write_test_image(tmp_path / "keep.png", size=(64, 64), color=(100, 120, 140))
    write_test_image(tmp_path / "dup.png", size=(64, 64), color=(100, 120, 140))
    (tmp_path / "keep.txt").write_text("tag", encoding="utf-8")
    (tmp_path / "dup.txt").write_text("tag", encoding="utf-8")

    removed = remove_duplicate_files(tmp_path, ["dup.png"])

    assert removed == 1
    assert (tmp_path / "keep.png").is_file()
    assert not (tmp_path / "dup.png").is_file()
    assert (tmp_path / "keep.txt").is_file()
    assert not (tmp_path / "dup.txt").is_file()


@pytest.mark.asyncio
async def test_remove_duplicates_via_service(storage_roots, datasets_service: DatasetsService) -> None:
    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    write_test_image(image_dir / "alpha.png", size=(64, 64), color=(100, 120, 140))
    write_test_image(image_dir / "beta.png", size=(64, 64), color=(100, 120, 140))
    (image_dir / "beta.png").write_bytes((image_dir / "alpha.png").read_bytes())

    dataset = await datasets_service.create_dataset(name="dupes", relative_path="images")
    scan = datasets_service.scan_duplicates(dataset)
    assert scan.duplicate_count == 1

    removed = await datasets_service.remove_duplicates(dataset)
    assert removed == 1
    assert (image_dir / "alpha.png").is_file()
    assert not (image_dir / "beta.png").is_file()

    scan_after = datasets_service.scan_duplicates(await datasets_service.get_dataset(dataset.id))  # type: ignore[arg-type]
    assert scan_after.duplicate_count == 0
