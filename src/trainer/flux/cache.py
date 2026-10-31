"""Latent and text-embedding caches for Flux training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

logger = logging.getLogger(__name__)

CacheProgressCallback = Callable[[int, int, str], None]

_PIXEL_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


def _fresh(source: Path, cached: Path) -> bool:
    return cached.is_file() and cached.stat().st_mtime >= source.stat().st_mtime


def build_flux_latent_cache(
    image_paths: list[Path],
    vae: nn.Module,
    device: torch.device,
    to_disk: bool,
    *,
    tag: str,
    multiple_of: int = 16,
    on_progress: Optional[CacheProgressCallback] = None,
    log: logging.Logger | None = None,
) -> dict[str, Tensor]:
    """Encode prepared images once. Returns raw (unscaled) latents on CPU, float32."""
    active_log = log or logger
    unique = list(dict.fromkeys(image_paths))
    active_log.info("Caching %s latents for %d unique images (to_disk=%s)...", tag, len(unique), to_disk)
    vae_dtype = next(vae.parameters()).dtype
    vae.eval()
    vae.to(device)

    cache: dict[str, Tensor] = {}
    for index, image_path in enumerate(unique, start=1):
        npz = image_path.parent / f"{image_path.stem}_{tag}.npz"
        if to_disk and _fresh(image_path, npz):
            cache[str(image_path)] = torch.from_numpy(np.load(npz)["latents"].copy())
        else:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            if width % multiple_of != 0 or height % multiple_of != 0:
                raise ValueError(
                    f"Prepared image {image_path.name} size {image.size} must be divisible by "
                    f"{multiple_of} for {tag}. Use a bucket/resolution step that is a multiple of {multiple_of}."
                )
            pixels = _PIXEL_TRANSFORM(image).unsqueeze(0).to(device, dtype=vae_dtype)
            with torch.no_grad():
                latent = vae.encode(pixels).latent_dist.sample().squeeze(0).float().cpu()
            if to_disk:
                np.savez(npz, latents=latent.numpy())
            cache[str(image_path)] = latent
        if on_progress is not None:
            on_progress(index, len(unique), "latents")

    vae.to("cpu")
    torch.cuda.empty_cache()
    return cache


def build_flux_text_cache(
    path_caption_pairs: list[tuple[Path, str]],
    encode: Callable[[str], tuple[Tensor, ...]],
    encoders: list[nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    to_disk: bool,
    *,
    tag: str,
    on_progress: Optional[CacheProgressCallback] = None,
    log: logging.Logger | None = None,
) -> dict[str, tuple[Tensor, ...]]:
    """Encode each unique caption once. ``encode`` returns CPU tensors with batch dim 1."""
    active_log = log or logger
    total = len({caption for _, caption in path_caption_pairs})
    active_log.info("Caching %s text embeddings for %d unique captions (to_disk=%s)...", tag, total, to_disk)
    for encoder in encoders:
        encoder.eval()
        encoder.to(device)

    cache: dict[str, tuple[Tensor, ...]] = {}
    processed = 0
    for image_path, caption in path_caption_pairs:
        if caption in cache:
            continue
        processed += 1
        disk_path = image_path.parent / f"{image_path.stem}_{tag}_te.pt"
        outputs: tuple[Tensor, ...] | None = None
        if to_disk and _fresh(image_path, disk_path):
            payload = torch.load(disk_path, map_location="cpu")
            if payload.get("tag") == tag and payload.get("caption") == caption:
                outputs = tuple(payload["tensors"])
        if outputs is None:
            with torch.no_grad():
                outputs = tuple(t.to(dtype=dtype).cpu() for t in encode(caption))
            if to_disk:
                torch.save({"tag": tag, "caption": caption, "tensors": list(outputs)}, disk_path)
        cache[caption] = outputs
        if on_progress is not None:
            on_progress(processed, total, "text_encoder")

    for encoder in encoders:
        encoder.to("cpu")
    torch.cuda.empty_cache()
    return cache
