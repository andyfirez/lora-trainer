"""Tests for ConceptDataset loading prepared images."""

from pathlib import Path

from PIL import Image

from src.trainer.config import ConceptConfig
from src.trainer.sdxl.dataset import ConceptDataset


def test_concept_dataset_loads_prepared_image_without_resize(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    prepared_dir = tmp_path / "images" / ".prepared" / "512"
    image_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)

    original = image_dir / "sample.png"
    Image.new("RGB", (800, 600), (200, 100, 50)).save(original)
    prepared = prepared_dir / "sample.png"
    Image.new("RGB", (512, 512), (10, 20, 30)).save(prepared)

    concept = ConceptConfig(
        dataset_id=1,
        image_dir=str(image_dir),
        prepared_dir=str(prepared_dir),
    )
    dataset = ConceptDataset(concept, resolution=512, cache_mode=False)
    item = dataset[0]

    assert item["pixel_values"].shape == (3, 512, 512)
    assert "caption" in item
