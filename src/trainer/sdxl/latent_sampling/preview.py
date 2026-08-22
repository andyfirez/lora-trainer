"""Cheap latent RGB previews written during denoising."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

LIVE_PREVIEW_FILENAME = "live_preview.jpg"


def latent_to_preview_image(latent: Tensor, *, width: int, height: int) -> Image.Image:
    """Map the first three latent channels to an RGB image (no VAE)."""
    sample = latent[0, :3].detach().float().cpu()
    sample = sample - sample.amin(dim=(1, 2), keepdim=True)
    denom = sample.amax(dim=(1, 2), keepdim=True).clamp_min(1e-5)
    sample = (sample / denom).clamp(0, 1)
    array = (sample.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    image = Image.fromarray(array, mode="RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image


def write_live_preview(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=72)


def should_write_preview(completed: int, total: int) -> bool:
    if completed <= 0 or total <= 0:
        return False
    interval = max(1, total // 15)
    return completed == total or completed % interval == 0


def preview_from_latent(
    latent: Tensor,
    *,
    path: Path,
    width: int,
    height: int,
    completed: int,
    total: int,
) -> None:
    if not should_write_preview(completed, total):
        return
    with torch.no_grad():
        image = latent_to_preview_image(latent, width=width, height=height)
    write_live_preview(path, image)
