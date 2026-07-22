"""Tests for legacy training YAML data migration helper."""

import pytest
import yaml
from src.db.migrations.training_yaml import migrate_training_yaml
from src.trainer.config import TrainConfig


def test_migrate_training_yaml_moves_legacy_learning_rate_to_parts() -> None:
    legacy_yaml = """
learning_rate: 0.0003
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
unet:
  train: true
text_encoder_1:
  train: true
  learning_rate: 0.00005
concepts:
  - dataset_id: 1
"""
    migrated = migrate_training_yaml(legacy_yaml)
    assert migrated is not None
    raw = yaml.safe_load(migrated) or {}
    assert "learning_rate" not in raw
    config = TrainConfig.from_yaml(migrated)
    assert config.unet.learning_rate == pytest.approx(3e-4)
    assert config.text_encoder_1.learning_rate == pytest.approx(5e-5)
    assert config.text_encoder_2.learning_rate == pytest.approx(3e-4)


def test_migrate_training_yaml_strips_inline_sample_prompts() -> None:
    legacy_yaml = """
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
concepts:
  - dataset_id: 1
sample_prompts:
  - test prompt
"""
    migrated = migrate_training_yaml(legacy_yaml)
    assert migrated is not None
    raw = yaml.safe_load(migrated) or {}
    assert "sample_prompts" not in raw


def test_migrate_training_yaml_removes_sample_after_training() -> None:
    legacy_yaml = """
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
sample_after_training: true
concepts:
  - dataset_id: 1
"""
    migrated = migrate_training_yaml(legacy_yaml)
    assert migrated is not None
    raw = yaml.safe_load(migrated) or {}
    assert "sample_after_training" not in raw


def test_migrate_training_yaml_returns_none_for_canonical_yaml() -> None:
    canonical = TrainConfig(
        concepts=[{"dataset_id": 1}],
    ).to_yaml()
    assert migrate_training_yaml(canonical) is None


def test_migrate_training_yaml_returns_none_for_invalid_yaml() -> None:
    assert migrate_training_yaml("not_a_valid_config: [[[") is None
