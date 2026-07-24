"""Tests for legacy sampling YAML data migration helper."""

import yaml

from src.db.migrations.sampling_yaml import migrate_sampling_yaml
from src.sampler.config import SamplingConfig


def test_migrate_sampling_yaml_moves_legacy_prompts_to_parameters() -> None:
    legacy_yaml = """
output_dir: /tmp/out
sample_prompts:
  - hello
sample_steps: 25
"""
    migrated = migrate_sampling_yaml(legacy_yaml)
    assert migrated is not None
    raw = yaml.safe_load(migrated) or {}
    assert "sample_prompts" not in raw
    assert "sample_steps" not in raw
    config = SamplingConfig.from_yaml(migrated)
    assert config.effective_prompts() == ["hello"]
    assert config.effective_steps() == 25


def test_migrate_sampling_yaml_returns_none_for_canonical_yaml() -> None:
    canonical = SamplingConfig(
        output_dir="/tmp/out",
        parameters={"prompt": {"mode": "fixed", "value": "hello"}},
    ).to_yaml()
    assert migrate_sampling_yaml(canonical) is None


def test_migrate_sampling_yaml_returns_none_for_invalid_yaml() -> None:
    assert migrate_sampling_yaml("not_a_valid_config: [[[") is None
