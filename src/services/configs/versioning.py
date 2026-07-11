"""Helpers for training config version history and lora_name suffixes."""

import re

import yaml
from src.trainer.config import TrainConfig

_LORA_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def strip_lora_version_suffix(name: str) -> str:
    return _LORA_VERSION_SUFFIX_RE.sub("", name)


def versioned_lora_name(lora_name: str, version: int) -> str:
    return f"{strip_lora_version_suffix(lora_name)}_v{version}"


def normalize_training_config_yaml(config_yaml: str) -> str:
    config = TrainConfig.from_yaml(config_yaml)
    base_name = strip_lora_version_suffix(config.lora_name)
    if config.lora_name == base_name:
        return config_yaml
    return config.model_copy(update={"lora_name": base_name}).to_yaml()


def apply_lora_version_to_train_config(
    config: TrainConfig,
    version: int | None,
) -> TrainConfig:
    base_name = strip_lora_version_suffix(config.lora_name)
    if version is None:
        return config.model_copy(update={"lora_name": base_name})
    return config.model_copy(update={"lora_name": versioned_lora_name(base_name, version)})


def canonical_yaml(yaml_str: str) -> str:
    data = yaml.safe_load(yaml_str) or {}
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=True)


def yaml_configs_equal(yaml_a: str, yaml_b: str) -> bool:
    config_a = TrainConfig.from_yaml(yaml_a)
    config_b = TrainConfig.from_yaml(yaml_b)
    normalized_a = config_a.model_copy(
        update={"lora_name": strip_lora_version_suffix(config_a.lora_name)}
    )
    normalized_b = config_b.model_copy(
        update={"lora_name": strip_lora_version_suffix(config_b.lora_name)}
    )
    return canonical_yaml(normalized_a.to_yaml()) == canonical_yaml(normalized_b.to_yaml())


def extract_lora_name(config_yaml: str) -> str | None:
    try:
        return strip_lora_version_suffix(TrainConfig.from_yaml(config_yaml).lora_name)
    except Exception:
        return None
