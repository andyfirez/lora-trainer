"""Unit tests for training config versioning helpers."""

from src.services.configs.versioning import (
    apply_lora_version_to_train_config,
    normalize_training_config_yaml,
    strip_lora_version_suffix,
    versioned_lora_name,
    yaml_configs_equal,
)
from src.trainer.config import TrainConfig


def test_strip_lora_version_suffix() -> None:
    assert strip_lora_version_suffix("MyLora_v3") == "MyLora"
    assert strip_lora_version_suffix("MyLora") == "MyLora"


def test_versioned_lora_name() -> None:
    assert versioned_lora_name("Winx_Bloom", 2) == "Winx_Bloom_v2"
    assert versioned_lora_name("Winx_Bloom_v1", 2) == "Winx_Bloom_v2"


def test_normalize_training_config_yaml() -> None:
    yaml_text = """base_model_name: x
lora_name: Winx_Bloom_v2
concepts: []
"""
    normalized = normalize_training_config_yaml(yaml_text)

    assert "lora_name: Winx_Bloom" in normalized
    assert "_v2" not in normalized


def test_apply_lora_version_to_train_config() -> None:
    config = TrainConfig(lora_name="Winx_Bloom")

    runtime = apply_lora_version_to_train_config(config, 3)

    assert runtime.lora_name == "Winx_Bloom_v3"


def test_yaml_configs_equal_ignores_key_order() -> None:
    yaml_a = "base_model_name: x\nepochs: 10\n"
    yaml_b = "epochs: 10\nbase_model_name: x\n"

    assert yaml_configs_equal(yaml_a, yaml_b)


def test_yaml_configs_equal_ignores_lora_name_version_suffix() -> None:
    yaml_a = "lora_name: demo\nepochs: 10\n"
    yaml_b = "lora_name: demo_v2\nepochs: 10\n"

    assert yaml_configs_equal(yaml_a, yaml_b)
