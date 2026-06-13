"""Dataset classes and helpers for SDXL LoRA training."""

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.trainer.config import ConceptConfig, TrainConfig

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_caption(image_path: Path, concept: ConceptConfig) -> str:
    caption_path = image_path.with_suffix(concept.caption_extension)
    caption = ""
    if caption_path.is_file():
        caption = caption_path.read_text(encoding="utf-8").strip()
    return f"{concept.caption_prefix}{caption}{concept.caption_suffix}"


def collect_all_image_paths_and_captions(
    config: TrainConfig,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return all unique image paths and all (image_path, caption) pairs across concepts."""
    all_paths: list[Path] = []
    all_pairs: list[tuple[Path, str]] = []

    for concept in config.concepts:
        image_dir = Path(concept.image_dir)
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
        for p in image_paths:
            caption = _load_caption(p, concept)
            all_paths.append(p)
            all_pairs.append((p, caption))

    return all_paths, all_pairs


def count_latent_cache_items(image_paths: list[Path]) -> int:
    return len(dict.fromkeys(image_paths))


def count_te_cache_items(path_caption_pairs: list[tuple[Path, str]]) -> int:
    seen: set[str] = set()
    count = 0
    for _, caption in path_caption_pairs:
        if caption in seen:
            continue
        seen.add(caption)
        count += 1
    return count


CacheProgressCallback = Callable[[int, int, str], None]


class ConceptDataset(Dataset):
    """Dataset for a single concept.

    In cache_mode=True, __getitem__ returns {"image_path": str, "caption": str}.
    In cache_mode=False, __getitem__ returns {"pixel_values": Tensor, "caption": str}.
    """

    def __init__(self, concept: ConceptConfig, resolution: int, cache_mode: bool = False) -> None:
        self._concept = concept
        image_dir = Path(concept.image_dir)
        self._image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
        self._cache_mode = cache_mode
        if not cache_mode:
            self._transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    @property
    def image_paths(self) -> list[Path]:
        return list(self._image_paths)

    def __len__(self) -> int:
        return len(self._image_paths) * self._concept.repeats

    def __getitem__(self, idx: int) -> dict:
        path = self._image_paths[idx % len(self._image_paths)]
        caption = _load_caption(path, self._concept)
        if self._cache_mode:
            return {"image_path": str(path), "caption": caption}
        image = Image.open(path).convert("RGB")
        return {"pixel_values": self._transform(image), "caption": caption}
