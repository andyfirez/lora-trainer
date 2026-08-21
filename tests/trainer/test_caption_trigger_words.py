"""Tests for trigger-word injection and training caption assembly."""

from src.trainer.config import ConceptConfig
from src.trainer.sdxl.caption import (
    apply_trigger_words_to_prompt,
    apply_trigger_words_to_sample_prompts,
    collect_trigger_words,
)
from src.trainer.sdxl.dataset import _build_caption


def test_collect_trigger_words_uses_first_concept_only() -> None:
    concepts = [
        ConceptConfig(dataset_id=1, trigger_words=["ohwx", "person"]),
        ConceptConfig(dataset_id=2, trigger_words=["style"]),
    ]
    assert collect_trigger_words(concepts) == ["ohwx", "person"]


def test_apply_trigger_words_to_prompt_prepends_words() -> None:
    assert apply_trigger_words_to_prompt("portrait", ["ohwx", "person"]) == "ohwx, person, portrait"


def test_apply_trigger_words_to_prompt_skips_when_already_present() -> None:
    prompt = "ohwx, person, portrait"
    assert apply_trigger_words_to_prompt(prompt, ["ohwx", "person"]) == prompt


def test_apply_trigger_words_to_prompt_handles_empty_prompt() -> None:
    assert apply_trigger_words_to_prompt("", ["ohwx"]) == "ohwx"


def test_apply_trigger_words_to_sample_prompts_noop_without_words() -> None:
    prompts = ["portrait", "full body"]
    assert apply_trigger_words_to_sample_prompts(prompts, []) == prompts


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
