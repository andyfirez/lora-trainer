"""Tests for sampling config GPU resolution."""

from src.sampler.config import SamplingConfig
from src.settings.models import GpuDefaultsSettings
from src.trainer.config import VaeDtype, WeightDtype


def test_sampling_config_entity_defaults_are_sparse() -> None:
    config = SamplingConfig()
    data = config._entity_data()

    assert "tf32" not in data
    assert "attention_mechanism" not in data
    assert "mixed_precision" not in data
    assert "vae_dtype" not in data
    assert "sample_vae_tiling" not in data


def test_sampling_config_to_inference_config_propagates_resolved_gpu(monkeypatch) -> None:
    from src.settings.app_settings import settings
    from src.settings.models import GpuDefaultsSettings

    monkeypatch.setattr(settings, "gpu_defaults", GpuDefaultsSettings())
    config = SamplingConfig(
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.FLOAT_16,
        sample_vae_tiling=False,
        sample_vae_fp32=True,
        sample_offload_unet_before_decode=False,
    )

    inference_config = config.to_inference_config()

    assert inference_config.mixed_precision == WeightDtype.FLOAT_16
    assert inference_config.vae_dtype == VaeDtype.FLOAT_16
    assert inference_config.tf32 is True
    assert inference_config.attention_mechanism == "sdpa"
    assert inference_config.sample_vae_tiling is False
    assert inference_config.sample_vae_fp32 is True
    assert inference_config.sample_offload_unet_before_decode is False


def test_sampling_config_inference_config_field_updates_uses_resolved_vae_tiling() -> None:
    config = SamplingConfig(sample_vae_tiling=False)

    updates = config.inference_config_field_updates()

    assert updates["sample_vae_tiling"] is False


def test_sampling_config_default_roundtrip() -> None:
    config = SamplingConfig.from_dict(SamplingConfig()._entity_data())
    assert config.parameters.lora_weight.first_value() == 1.0


def test_sampling_config_inference_config_field_updates_preserves_prompts() -> None:
    config = SamplingConfig.from_dict(
        {
            "output_dir": "/tmp",
            "parameters": {"prompt": {"mode": "vary", "values": ["a", "b"]}},
        }
    )
    updates = config.inference_config_field_updates()
    assert updates["sample_prompts"] == ["a", "b"]


def test_sampling_config_snapshot_includes_gpu_fields() -> None:
    defaults = GpuDefaultsSettings(tf32=False, attention_mechanism="xformers")
    config = SamplingConfig(mixed_precision=WeightDtype.BFLOAT_16).with_resolved_gpu(defaults)
    snapshot = config.to_snapshot()

    assert snapshot["tf32"] is False
    assert snapshot["attention_mechanism"] == "xformers"
    assert snapshot["mixed_precision"] == "bfloat16"
