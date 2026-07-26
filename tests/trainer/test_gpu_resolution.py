from src.settings.models import GpuDefaultsSettings
from src.trainer.config import VaeDtype, WeightDtype
from src.trainer.gpu_resolution import (
    resolve_gpu_config,
    strip_global_gpu_keys,
    strip_gpu_overrides_matching_defaults,
)


def test_resolve_gpu_config_uses_defaults() -> None:
    defaults = GpuDefaultsSettings(
        tf32=True,
        attention_mechanism="sdpa",
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.AUTO,
        sample_vae_tiling=True,
    )
    resolved = resolve_gpu_config(defaults=defaults)
    assert resolved.tf32 is True
    assert resolved.attention_mechanism == "sdpa"
    assert resolved.mixed_precision == WeightDtype.FLOAT_16
    assert resolved.vae_dtype == VaeDtype.AUTO
    assert resolved.sample_vae_tiling is True


def test_resolve_gpu_config_applies_overrides() -> None:
    defaults = GpuDefaultsSettings()
    resolved = resolve_gpu_config(
        defaults=defaults,
        mixed_precision=WeightDtype.BFLOAT_16,
        vae_dtype=VaeDtype.FLOAT_16,
        sample_vae_tiling=False,
    )
    assert resolved.mixed_precision == WeightDtype.BFLOAT_16
    assert resolved.vae_dtype == VaeDtype.FLOAT_16
    assert resolved.sample_vae_tiling is False
    assert resolved.tf32 == defaults.tf32
    assert resolved.attention_mechanism == defaults.attention_mechanism


def test_strip_global_gpu_keys() -> None:
    data = {"tf32": False, "attention_mechanism": "xformers", "mixed_precision": "float16"}
    stripped = strip_global_gpu_keys(data)
    assert stripped == {"mixed_precision": "float16"}


def test_strip_gpu_overrides_matching_defaults() -> None:
    defaults = GpuDefaultsSettings(
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.AUTO,
        sample_vae_tiling=True,
    )
    data = {
        "mixed_precision": "float16",
        "vae_dtype": "auto",
        "sample_vae_tiling": True,
        "lora_rank": 32,
    }
    stripped = strip_gpu_overrides_matching_defaults(data, defaults)
    assert stripped == {"lora_rank": 32}
