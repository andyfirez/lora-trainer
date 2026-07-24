"""Tests for legacy sampling config migration."""

import yaml

from src.db.migrations.sampling_yaml import migrate_sampling_yaml, migrate_sampling_yaml_raw
from src.sampler.config import SamplingConfig
from src.sampler.sweep.models import SweepMode
from src.services.configs.versioning import normalize_sampling_config_yaml


def test_legacy_prompts_migrate_to_parameters() -> None:
    migrated = migrate_sampling_yaml(
        "output_dir: /tmp/out\nsample_prompts:\n  - hello\n  - world\nsample_steps: 20\n"
    )
    assert migrated is not None
    config = SamplingConfig.from_yaml(migrated)
    assert config.effective_prompts() == ["hello", "world"]
    assert config.effective_steps() == 20
    assert config.parameters.prompt.mode == SweepMode.VARY
    assert config.parameters.steps.first_value() == 20
    raw = yaml.safe_load(migrated) or {}
    assert "sample_prompts" not in raw
    assert "sample_steps" not in raw


def test_conflict_parameters_win_over_legacy() -> None:
    legacy_yaml = """
output_dir: /tmp/out
base_model_name: D:/models/legacy.safetensors
sample_prompts:
  - legacy prompt
parameters:
  base_model_name:
    mode: fixed
    value: D:/models/parameters.safetensors
  prompt:
    mode: fixed
    value: parameters prompt
"""
    migrated = migrate_sampling_yaml(legacy_yaml)
    assert migrated is not None
    config = SamplingConfig.from_yaml(migrated)
    assert config.effective_base_model_name() == "D:/models/parameters.safetensors"
    assert config.effective_prompts() == ["parameters prompt"]
    raw = yaml.safe_load(migrated) or {}
    assert "base_model_name" not in raw
    assert "sample_prompts" not in raw


def test_strips_deprecated_sampling_keys() -> None:
    legacy_yaml = """
output_dir: /tmp/out
source_type: manual
use_reforge_sampler: false
sample_sampler: euler_a
parameters:
  prompt:
    mode: fixed
    value: test
"""
    migrated = migrate_sampling_yaml(legacy_yaml)
    assert migrated is not None
    raw = yaml.safe_load(migrated) or {}
    assert "source_type" not in raw
    assert "use_reforge_sampler" not in raw
    assert "sample_sampler" not in raw


def test_build_sampling_field_updates_preserves_prompts() -> None:
    config = SamplingConfig.from_yaml(
        normalize_sampling_config_yaml("output_dir: /tmp\nsample_prompts:\n  - a\n  - b\n")
    )
    updates = config.build_sampling_field_updates()
    assert updates["sample_prompts"] == ["a", "b"]


def test_new_format_yaml_roundtrip() -> None:
    yaml_text = SamplingConfig.default_yaml()
    config = SamplingConfig.from_yaml(yaml_text)
    assert config.parameters.lora_weight.first_value() == 1.0


def test_migrate_sampling_yaml_returns_none_for_canonical_yaml() -> None:
    canonical = SamplingConfig(
        output_dir="/tmp/out",
        parameters={"prompt": {"mode": "fixed", "value": "hello"}},
    ).to_yaml()
    assert migrate_sampling_yaml(canonical) is None


def test_migrate_sampling_yaml_raw_syncs_lora_paths() -> None:
    raw = migrate_sampling_yaml_raw(
        {
            "output_dir": "/tmp/out",
            "parameters": {
                "prompt": {"mode": "fixed", "value": "test"},
                "lora_path": {
                    "mode": "fixed",
                    "value": {"path": "D:/loras/a.safetensors", "trigger": "a"},
                },
            },
        }
    )
    assert raw["lora_paths"] == ["D:/loras/a.safetensors"]
