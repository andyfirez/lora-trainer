"""Tests for per-LoRA trigger prompt assembly."""

from src.sampler.sweep.models import parse_trigger_words
from src.trainer.sdxl.caption import apply_trigger_words_to_prompt


def test_apply_lora_trigger_to_prompt() -> None:
    prompt = apply_trigger_words_to_prompt("portrait", parse_trigger_words("ohwx, person"))
    assert prompt == "ohwx, person, portrait"


def test_apply_lora_trigger_skips_empty() -> None:
    prompt = apply_trigger_words_to_prompt("portrait", parse_trigger_words(""))
    assert prompt == "portrait"
