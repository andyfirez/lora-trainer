"""Unit tests for training config normalization helpers."""

from src.services.configs.versioning import (
    normalize_training_config_yaml,
    strip_lora_version_suffix,
    yaml_configs_equal,
)


def test_strip_lora_version_suffix() -> None:
    assert strip_lora_version_suffix("MyLora_v3") == "MyLora"
    assert strip_lora_version_suffix("MyLora") == "MyLora"


def test_normalize_training_config_yaml() -> None:
    yaml_in = """base_model_name: x
lora_name: Winx_Bloom_v2
"""
    normalized = normalize_training_config_yaml(yaml_in)
    assert "lora_name: Winx_Bloom" in normalized


def test_yaml_configs_equal_ignores_lora_name_version_suffix() -> None:
    yaml_a = "lora_name: demo\nepochs: 10\n"
    yaml_b = "lora_name: demo_v2\nepochs: 10\n"
    assert yaml_configs_equal(yaml_a, yaml_b)
