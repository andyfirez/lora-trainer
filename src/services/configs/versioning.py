"""Helpers for training and sampling config normalization."""

import re

import yaml
from src.db.migrations.sampling_yaml import migrate_sampling_yaml_raw
from src.sampler.config import SamplingConfig
from src.trainer.config import TrainConfig

_LORA_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def strip_lora_version_suffix(name: str) -> str:
    return _LORA_VERSION_SUFFIX_RE.sub("", name)


def normalize_training_config_yaml(config_yaml: str) -> str:
    from src.db.migrations.training_yaml import migrate_training_yaml_raw

    data = yaml.safe_load(config_yaml) or {}
    if not isinstance(data, dict):
        data = {}
    migrated_data = migrate_training_yaml_raw(data)
    prepared = yaml.dump(migrated_data, allow_unicode=True, sort_keys=False)
    config = TrainConfig.from_yaml(prepared)
    base_name = strip_lora_version_suffix(config.lora_name)
    if config.lora_name != base_name:
        config = config.model_copy(update={"lora_name": base_name})
    return config.to_yaml()


def normalize_sampling_config_yaml(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml) or {}
    if not isinstance(data, dict):
        data = {}
    migrated_data = migrate_sampling_yaml_raw(data)
    return SamplingConfig.model_validate(migrated_data).to_yaml()


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
