"""Tests for training caption assembly with trigger words."""

from src.trainer.config import ConceptConfig
from src.trainer.sdxl.dataset import _build_caption


def test_build_caption_with_trigger_words_and_tags() -> None:
    concept = ConceptConfig(dataset_id=1, trigger_words=["ohwx", "person"])
    assert _build_caption("1girl, smile", concept) == "ohwx, person, 1girl, smile"


def test_build_caption_without_trigger_words() -> None:
    concept = ConceptConfig(dataset_id=1)
    assert _build_caption("1girl, smile", concept) == "1girl, smile"


def test_build_caption_without_tags() -> None:
    concept = ConceptConfig(dataset_id=1, trigger_words=["ohwx", "person"])
    assert _build_caption("", concept) == "ohwx, person"


def test_build_caption_with_suffix() -> None:
    concept = ConceptConfig(dataset_id=1, trigger_words=["ohwx"], caption_suffix=" END")
    assert _build_caption("smile", concept) == "ohwx, smile END"
