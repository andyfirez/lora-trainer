"""Pre-compute and cache VAE latents for SDXL LoRA training."""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

logger = logging.getLogger(__name__)

_NPZ_SUFFIX = "_sdxl.npz"
CacheProgressCallback = Callable[[int, int, str], None]


def _npz_path(image_path: Path) -> Path:
    return image_path.parent / (image_path.stem + _NPZ_SUFFIX)


def _disk_cache_valid(image_path: Path, npz: Path) -> bool:
    return npz.is_file() and npz.stat().st_mtime >= image_path.stat().st_mtime


def _make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def build_latent_cache(
    image_paths: list[Path],
    resolution: int,
    vae: torch.nn.Module,
    device: torch.device,
    to_disk: bool,
    on_progress: Optional[CacheProgressCallback] = None,
    log: logging.Logger | None = None,
) -> dict[str, Tensor]:
    """Encode all prepared images through VAE once and cache the resulting latents."""
    transform = _make_transform()
    cache: dict[str, Tensor] = {}
    active_log = log or logger

    unique_paths = list(dict.fromkeys(image_paths))
    active_log.info("Caching latents for %d unique images (to_disk=%s)...", len(unique_paths), to_disk)

    vae.eval()
    vae.to(device)

    loaded_from_disk = 0
    encoded = 0

    for index, image_path in enumerate(unique_paths, start=1):
        npz = _npz_path(image_path)

        if to_disk and _disk_cache_valid(image_path, npz):
            data = np.load(npz)
            latent = torch.from_numpy(data["latents"].copy())
            cache[str(image_path)] = latent
            loaded_from_disk += 1
            if on_progress is not None:
                on_progress(index, len(unique_paths), "latents")
            continue

        image = Image.open(image_path).convert("RGB")
        if image.size != (resolution, resolution):
            raise ValueError(
                f"Prepared image {image_path.name} has size {image.size}, "
                f"expected ({resolution}, {resolution})"
            )
        pixel_values = transform(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            latent = vae.encode(pixel_values).latent_dist.sample()
            latent = (latent * vae.config.scaling_factor).squeeze(0).cpu()

        if to_disk:
            np.savez(npz, latents=latent.float().numpy())

        cache[str(image_path)] = latent
        encoded += 1
        if on_progress is not None:
            on_progress(index, len(unique_paths), "latents")

    active_log.info(
        "Latent cache ready: %d encoded, %d loaded from disk. Moving VAE to CPU.",
        encoded,
        loaded_from_disk,
    )
    vae.to("cpu")
    torch.cuda.empty_cache()

    return cache
