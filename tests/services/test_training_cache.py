"""Tests for training disk cache invalidation."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.services.datasets.preprocess import prepared_dir_path
from src.services.datasets.training_cache import invalidate_te_cache_for_image


def _write_image(path: Path, size: tuple[int, int] = (1024, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 100, 100)).save(path)


def _write_te_cache(prepared_dir: Path, stem: str) -> Path:
    cache_path = prepared_dir / f"{stem}_te.npz"
    np.savez(
        cache_path,
        prompt_embeds=np.zeros((1, 77, 2048), dtype=np.float32),
        pooled_prompt_embeds=np.zeros((1, 1280), dtype=np.float32),
    )
    return cache_path


def test_invalidate_te_cache_deletes_npz(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    resolution = 1024
    _write_image(image_dir / "img.png")
    prepared_dir = prepared_dir_path(image_dir, resolution)
    _write_image(prepared_dir / "img.png")
    cache_path = _write_te_cache(prepared_dir, "img")

    invalidate_te_cache_for_image(image_dir, "img.png", resolution)

    assert not cache_path.is_file()


def test_invalidate_te_cache_without_resolution_is_noop(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    prepared_dir = prepared_dir_path(image_dir, 1024)
    prepared_dir.mkdir(parents=True)
    cache_path = _write_te_cache(prepared_dir, "img")

    invalidate_te_cache_for_image(image_dir, "img.png", None)

    assert cache_path.is_file()


def test_invalidate_te_cache_without_prepared_image_is_noop(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    prepared_dir = prepared_dir_path(image_dir, 1024)
    prepared_dir.mkdir(parents=True)
    cache_path = _write_te_cache(prepared_dir, "img")

    invalidate_te_cache_for_image(image_dir, "missing.png", 1024)

    assert cache_path.is_file()


def test_invalidate_te_cache_uses_alt_png_prepared_path(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    resolution = 1024
    _write_image(image_dir / "photo.jpeg")
    prepared_dir = prepared_dir_path(image_dir, resolution)
    _write_image(prepared_dir / "photo.png")
    cache_path = _write_te_cache(prepared_dir, "photo")

    invalidate_te_cache_for_image(image_dir, "photo.jpeg", resolution)

    assert not cache_path.is_file()
