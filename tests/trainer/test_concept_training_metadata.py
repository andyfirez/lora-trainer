"""Tests for concept training metadata helpers."""

from src.trainer.concept_training_metadata import (
    ConceptTrainingMetadata,
    ImageTrainingMeta,
    resolve_reference_add_time_ids,
)


def _concept_meta(
    dataset_id: int,
    entries: dict[str, tuple[int, int, int, int, int, int]],
) -> ConceptTrainingMetadata:
    by_filename = {
        filename: ImageTrainingMeta(
            filename=filename,
            add_time_ids=add_time_ids,
            bucket_width=add_time_ids[5],
            bucket_height=add_time_ids[4],
        )
        for filename, add_time_ids in entries.items()
    }
    return ConceptTrainingMetadata(
        dataset_id=dataset_id,
        enable_bucket=True,
        by_filename=by_filename,
    )


def test_resolve_reference_add_time_ids_median_for_matching_bucket() -> None:
    metadata = {
        1: _concept_meta(
            1,
            {
                "a.jpg": (918, 1216, 3, 0, 1024, 768),
                "b.jpg": (920, 1216, 5, 0, 1024, 768),
                "c.jpg": (900, 1200, 0, 0, 1024, 1024),
            },
        ),
    }
    result = resolve_reference_add_time_ids(metadata, dataset_ids=[1], width=768, height=1024)
    assert result == (919.0, 1216.0, 4.0, 0.0, 1024.0, 768.0)


def test_resolve_reference_add_time_ids_no_matching_bucket_returns_none() -> None:
    metadata = {
        1: _concept_meta(1, {"a.jpg": (918, 1216, 3, 0, 1024, 768)}),
    }
    assert resolve_reference_add_time_ids(metadata, dataset_ids=[1], width=1024, height=1024) is None


def test_resolve_reference_add_time_ids_merges_multiple_datasets() -> None:
    metadata = {
        1: _concept_meta(1, {"a.jpg": (918, 1216, 3, 0, 1024, 768)}),
        2: _concept_meta(2, {"b.jpg": (920, 1220, 7, 0, 1024, 768)}),
    }
    result = resolve_reference_add_time_ids(metadata, dataset_ids=[1, 2], width=768, height=1024)
    assert result == (919.0, 1218.0, 5.0, 0.0, 1024.0, 768.0)


def test_resolve_reference_add_time_ids_missing_dataset_skipped() -> None:
    metadata = {
        1: _concept_meta(1, {"a.jpg": (918, 1216, 3, 0, 1024, 768)}),
    }
    result = resolve_reference_add_time_ids(metadata, dataset_ids=[1, 99], width=768, height=1024)
    assert result == (918.0, 1216.0, 3.0, 0.0, 1024.0, 768.0)
