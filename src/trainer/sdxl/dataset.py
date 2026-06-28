"""Dataset classes and helpers for SDXL LoRA training."""

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.trainer.config import ConceptConfig, TrainConfig
from src.trainer.sdxl.caption import join_trigger_words_and_text

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _concept_image_dir(concept: ConceptConfig) -> Path:
    if not concept.image_dir:
        raise ValueError(f"Concept with dataset_id={concept.dataset_id} has no resolved image_dir")
    return Path(concept.image_dir)


def _concept_prepared_dir(concept: ConceptConfig) -> Path:
    if not concept.prepared_dir:
        raise ValueError(f"Concept with dataset_id={concept.dataset_id} has no resolved prepared_dir")
    return Path(concept.prepared_dir)


def _build_caption(raw: str, concept: ConceptConfig) -> str:
    return join_trigger_words_and_text(concept.trigger_words, raw, concept.caption_suffix)


def _load_caption(image_path: Path, concept: ConceptConfig) -> str:
    caption_path = image_path.with_suffix(concept.caption_extension)
    raw = ""
    if caption_path.is_file():
        raw = caption_path.read_text(encoding="utf-8").strip()
    return _build_caption(raw, concept)


def collect_all_image_paths_and_captions(
    config: TrainConfig,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return all unique prepared image paths and all (prepared_path, caption) pairs."""
    all_paths: list[Path] = []
    all_pairs: list[tuple[Path, str]] = []

    for concept in config.concepts:
        prepared_dir = _concept_prepared_dir(concept)
        image_dir = _concept_image_dir(concept)
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
        for original_path in image_paths:
            prepared_path = prepared_dir / original_path.name
            if not prepared_path.is_file():
                alt_png = prepared_dir / f"{original_path.stem}.png"
                if alt_png.is_file():
                    prepared_path = alt_png
            caption = _load_caption(original_path, concept)
            all_paths.append(prepared_path)
            all_pairs.append((prepared_path, caption))

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
        self._resolution = resolution
        image_dir = _concept_image_dir(concept)
        prepared_dir = _concept_prepared_dir(concept)
        self._image_dir = image_dir
        self._prepared_dir = prepared_dir
        self._image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
        self._cache_mode = cache_mode
        if not cache_mode:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    @property
    def image_paths(self) -> list[Path]:
        return [self._prepared_path(p) for p in self._image_paths]

    def _prepared_path(self, original_path: Path) -> Path:
        prepared = self._prepared_dir / original_path.name
        if prepared.is_file():
            return prepared
        alt_png = self._prepared_dir / f"{original_path.stem}.png"
        if alt_png.is_file():
            return alt_png
        return prepared

    def __len__(self) -> int:
        return len(self._image_paths) * self._concept.repeats

    def __getitem__(self, idx: int) -> dict:
        original_path = self._image_paths[idx % len(self._image_paths)]
        prepared_path = self._prepared_path(original_path)
        caption = _load_caption(original_path, self._concept)
        if self._cache_mode:
            return {"image_path": str(prepared_path), "caption": caption}
        image = Image.open(prepared_path).convert("RGB")
        if image.size != (self._resolution, self._resolution):
            raise ValueError(
                f"Prepared image {prepared_path.name} has size {image.size}, "
                f"expected ({self._resolution}, {self._resolution})"
            )
        return {"pixel_values": self._transform(image), "caption": caption}
