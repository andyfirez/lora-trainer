"""Tests for training concept dataset resolution."""

import pytest

from src.db.repositories.dataset_repo import DatasetRepository
from src.services.datasets.exceptions import DatasetNotFoundError
from src.trainer.concept_resolution import resolve_dataset_ids
from src.trainer.config import ConceptConfig, TrainConfig


def test_resolve_concepts_sets_image_dir() -> None:
    config = TrainConfig(
        concepts=[
            ConceptConfig(dataset_id=1),
            ConceptConfig(dataset_id=2, repeats=3),
        ],
    )
    resolved = config.resolve_concepts({1: "/data/a", 2: "/data/b"})

    assert resolved.concepts[0].image_dir == "/data/a"
    assert resolved.concepts[1].image_dir == "/data/b"
    assert resolved.concepts[1].repeats == 3


def test_resolve_concepts_raises_for_missing_dataset() -> None:
    config = TrainConfig(concepts=[ConceptConfig(dataset_id=42)])

    with pytest.raises(ValueError, match="Dataset with id=42 not found"):
        config.resolve_concepts({})


def test_to_yaml_excludes_resolved_image_dir() -> None:
    config = TrainConfig(
        concepts=[ConceptConfig(dataset_id=1, image_dir="/resolved/path")],
    )
    yaml_text = config.to_yaml()

    assert "dataset_id" in yaml_text
    assert "image_dir" not in yaml_text


@pytest.mark.asyncio
async def test_resolve_dataset_ids(session, datasets_service, tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dataset = await datasets_service.create_dataset(name="demo", image_dir=str(image_dir))
    repo = DatasetRepository(session)

    result = await resolve_dataset_ids([dataset.id], repo)

    assert result[dataset.id] == str(image_dir)


@pytest.mark.asyncio
async def test_resolve_dataset_ids_raises_when_missing(session) -> None:
    repo = DatasetRepository(session)

    with pytest.raises(DatasetNotFoundError):
        await resolve_dataset_ids([999], repo)
