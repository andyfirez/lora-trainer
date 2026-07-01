"""Tests for ConceptDataset loading prepared images."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.trainer.config import ConceptConfig
from src.trainer.concept_training_metadata import ImageTrainingMeta
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
    expected = torch.tensor((512, 512, 0, 0, 512, 512), dtype=torch.float32)
    assert torch.equal(item["add_time_ids"], expected)
    assert "caption" in item


def test_dataloader_collate_add_time_ids_without_cross_sample_mixing(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    prepared_dir = tmp_path / "images" / ".prepared" / "512"
    image_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)

    first_original = image_dir / "first.png"
    second_original = image_dir / "second.png"
    Image.new("RGB", (800, 600), (200, 100, 50)).save(first_original)
    Image.new("RGB", (900, 700), (50, 100, 200)).save(second_original)
    Image.new("RGB", (1024, 768), (10, 20, 30)).save(prepared_dir / "first.png")
    Image.new("RGB", (768, 512), (30, 20, 10)).save(prepared_dir / "second.png")

    first_meta = ImageTrainingMeta(
        filename="first.png",
        add_time_ids=(918, 1216, 3, 0, 768, 1024),
        bucket_width=1024,
        bucket_height=768,
    )
    second_meta = ImageTrainingMeta(
        filename="second.png",
        add_time_ids=(700, 900, 0, 5, 512, 768),
        bucket_width=768,
        bucket_height=512,
    )
    concept = ConceptConfig(
        dataset_id=1,
        image_dir=str(image_dir),
        prepared_dir=str(prepared_dir),
    )
    dataset = ConceptDataset(
        concept,
        resolution=512,
        cache_mode=True,
        enable_bucket=True,
        image_meta={
            "first.png": first_meta,
            "second.png": second_meta,
        },
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert batch["add_time_ids"].shape == (2, 6)
    assert torch.equal(
        batch["add_time_ids"][0],
        torch.tensor(first_meta.add_time_ids, dtype=torch.float32),
    )
    assert torch.equal(
        batch["add_time_ids"][1],
        torch.tensor(second_meta.add_time_ids, dtype=torch.float32),
    )
