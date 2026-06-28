import pytest

from src.sampler.config import SamplingConfig
from src.trainer.config import VaeDtype, WeightDtype


def test_sampling_config_defaults_include_performance_fields() -> None:
    config = SamplingConfig()

    assert config.attention_mechanism == "sdpa"
    assert config.mixed_precision == WeightDtype.FLOAT_16
    assert config.vae_dtype == VaeDtype.AUTO
    assert config.tf32 is True


def test_sampling_config_to_train_config_propagates_performance_fields() -> None:
    config = SamplingConfig(
        attention_mechanism="sdpa",
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.FLOAT_16,
        tf32=False,
    )

    train_config = config.to_train_config()

    assert train_config.attention_mechanism == "sdpa"
    assert train_config.mixed_precision == WeightDtype.FLOAT_16
    assert train_config.vae_dtype == VaeDtype.FLOAT_16
    assert train_config.tf32 is False
    assert train_config.sample_prompts == config.sample_prompts
