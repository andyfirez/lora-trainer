"""Tests for legacy sampling config migration."""

from src.sampler.config import SamplingConfig
from src.sampler.sweep.models import SweepMode


def test_legacy_prompts_migrate_to_parameters() -> None:
    config = SamplingConfig.from_yaml(
        "sample_prompts:\n  - hello\n  - world\nsample_steps: 20\n"
    )
    assert config.effective_prompts() == ["hello", "world"]
    assert config.sample_steps == 20
    assert config.parameters.prompt.mode == SweepMode.VARY
    assert config.parameters.steps.first_value() == 20


def test_build_sampling_field_updates_preserves_prompts() -> None:
    config = SamplingConfig.from_yaml("sample_prompts:\n  - a\n  - b\n")
    updates = config.build_sampling_field_updates()
    assert updates["sample_prompts"] == ["a", "b"]


def test_new_format_yaml_roundtrip() -> None:
    yaml_text = SamplingConfig.default_yaml()
    config = SamplingConfig.from_yaml(yaml_text)
    assert config.source_type == "manual"
    assert config.parameters.lora_weight.first_value() == 1.0
