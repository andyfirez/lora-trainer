from pathlib import Path

import torch
from PIL import Image
from src.trainer.sdxl.latent_sampling.preview import (
    LIVE_PREVIEW_FILENAME,
    latent_to_preview_image,
    preview_from_latent,
    should_write_preview,
    write_live_preview,
)


def test_should_write_preview_includes_last_step() -> None:
    assert should_write_preview(0, 30) is False
    assert should_write_preview(30, 30) is True
    assert should_write_preview(2, 30) is True
    assert should_write_preview(1, 30) is False


def test_latent_to_preview_image_resizes() -> None:
    latent = torch.zeros(1, 4, 8, 8)
    latent[0, 0] = 1.0
    image = latent_to_preview_image(latent, width=64, height=48)
    assert image.size == (64, 48)
    assert image.mode == "RGB"


def test_write_live_preview_replaces_file(tmp_path: Path) -> None:
    path = tmp_path / LIVE_PREVIEW_FILENAME
    write_live_preview(path, Image.new("RGB", (8, 8), color="red"))
    write_live_preview(path, Image.new("RGB", (8, 8), color="blue"))
    assert path.is_file()


def test_preview_from_latent_skips_unwanted_steps(tmp_path: Path) -> None:
    path = tmp_path / LIVE_PREVIEW_FILENAME
    latent = torch.rand(1, 4, 8, 8)
    preview_from_latent(latent, path=path, width=16, height=16, completed=1, total=30)
    assert not path.exists()
    preview_from_latent(latent, path=path, width=16, height=16, completed=30, total=30)
    assert path.is_file()
