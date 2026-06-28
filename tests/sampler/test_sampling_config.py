import pytest

from src.sampler.config import SamplingConfig
from src.trainer.config import VaeDtype, WeightDtype


def test_sampling_config_defaults_include_performance_fields() -> None:
    config = SamplingConfig()

    assert config.attention_mechanism == "sdpa"
    assert config.mixed_precision == WeightDtype.FLOAT_16
    assert config.vae_dtype == VaeDtype.AUTO
    assert config.tf32 is True
    assert config.sample_vae_tiling is True
    assert config.sample_vae_fp32 is False
    assert config.sample_offload_unet_before_decode is True


def test_sampling_config_to_train_config_propagates_performance_fields() -> None:
    config = SamplingConfig(
        attention_mechanism="sdpa",
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.FLOAT_16,
        tf32=False,
        sample_vae_tiling=False,
        sample_vae_fp32=True,
        sample_offload_unet_before_decode=False,
    )

    train_config = config.to_train_config()

    assert train_config.attention_mechanism == "sdpa"
    assert train_config.mixed_precision == WeightDtype.FLOAT_16
    assert train_config.vae_dtype == VaeDtype.FLOAT_16
    assert train_config.tf32 is False
    assert train_config.sample_prompts == config.sample_prompts
    assert train_config.sample_vae_tiling is False
    assert train_config.sample_vae_fp32 is True
    assert train_config.sample_offload_unet_before_decode is False


def test_sampling_config_build_sampling_field_updates_includes_vae_decode_fields() -> None:
    config = SamplingConfig(
        sample_vae_tiling=False,
        sample_vae_fp32=True,
        sample_offload_unet_before_decode=False,
    )

    updates = config.build_sampling_field_updates()

    assert updates["sample_vae_tiling"] is False
    assert updates["sample_vae_fp32"] is True
    assert updates["sample_offload_unet_before_decode"] is False
