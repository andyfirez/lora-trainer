"""Tests for <lora:name:weight> prompt parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sampler.experiment.models import ExperimentParameters
from src.sampler.experiment.combinations import build_combinations
from src.sampler.lora_prompt import (
    apply_prompt_lora_to_params,
    parse_lora_tags,
    resolve_lora_tags,
)
from src.services.storage.inference_loras import discover_inference_loras


def test_parse_lora_tags_strips_single_tag() -> None:
    cleaned, tags = parse_lora_tags("a portrait <lora:style:0.8> detailed")
    assert cleaned == "a portrait detailed"
    assert len(tags) == 1
    assert tags[0].name == "style"
    assert tags[0].weight == 0.8


def test_parse_lora_tags_default_weight() -> None:
    _, tags = parse_lora_tags("<lora:foo> subject")
    assert tags[0].name == "foo"
    assert tags[0].weight == 1.0


def test_parse_lora_tags_multiple_and_case_insensitive() -> None:
    cleaned, tags = parse_lora_tags("<LoRA:a:1> x <LORA:b:0.5>")
    assert cleaned == "x"
    assert [(t.name, t.weight) for t in tags] == [("a", 1.0), ("b", 0.5)]


def test_resolve_lora_tags_uses_inference_catalog(storage_roots) -> None:
    inference_root: Path = storage_roots["inference_lora"]
    weights = inference_root / "my_style.safetensors"
    weights.write_bytes(b"placeholder")

    entries = resolve_lora_tags(parse_lora_tags("<lora:my_style:0.7>")[1])
    assert len(entries) == 1
    assert entries[0].path == "my_style.safetensors"
    assert entries[0].weight == 0.7


def test_apply_prompt_lora_to_params_overrides_lora_path() -> None:
    params = {
        "prompt": "hello <lora:test:0.5>",
        "lora_path": "old.safetensors",
        "lora_weight": 1.0,
    }
    with pytest.raises(ValueError):
        apply_prompt_lora_to_params(params)


def test_build_combinations_applies_prompt_lora_tags(storage_roots) -> None:
    inference_root: Path = storage_roots["inference_lora"]
    (inference_root / "cat.safetensors").write_bytes(b"x")

    parameters = ExperimentParameters.model_validate(
        {
            "prompt": {"mode": "fixed", "value": "photo <lora:cat:0.9>"},
            "lora_path": {"mode": "fixed", "value": {"path": None, "trigger": ""}},
            "lora_weight": {"mode": "fixed", "value": 1.0},
            "steps": {"mode": "fixed", "value": 20},
            "cfg_scale": {"mode": "fixed", "value": 7.0},
            "seed": {"mode": "fixed", "value": 1},
            "base_model_name": {"mode": "fixed", "value": "test-model"},
            "negative_prompt": {"mode": "fixed", "value": ""},
            "width": {"mode": "fixed", "value": 512},
            "height": {"mode": "fixed", "value": 512},
            "scheduler": {"mode": "fixed", "value": "euler_a"},
        }
    )
    combos = build_combinations(parameters)
    assert len(combos) == 1
    assert combos[0].params["prompt"] == "photo"
    assert combos[0].params["lora_path"] == "cat.safetensors"
    assert combos[0].params["infotext_prompt"] == "photo <lora:cat:0.9>"


def test_discover_inference_loras_nested_and_collision(storage_roots) -> None:
    inference_root: Path = storage_roots["inference_lora"]
    nested = inference_root / "nested"
    nested.mkdir()
    (nested / "dup.safetensors").write_bytes(b"a")
    (inference_root / "dup.safetensors").write_bytes(b"b")

    loras = discover_inference_loras()
    names = {entry.name for entry in loras}
    assert "dup" in names
    assert len([entry for entry in loras if entry.name == "dup"]) == 1
