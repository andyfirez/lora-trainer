import pytest
import yaml

from src.trainer.config import (
    Flux1Variant,
    FluxQuantization,
    ModelType,
    TrainConfig,
    WeightDtype,
)
from src.trainer.gpu_config_validation import validate_gpu_config
from src.trainer.config import VaeDtype


def test_legacy_yaml_without_model_type_is_sdxl():
    config = TrainConfig.from_yaml("base_model_name: model.safetensors\nlora_rank: 8\n")
    assert config.model_type == ModelType.SDXL
    assert config.flux1 is None
    assert config.flux2_klein is None
    assert config.unet.weight_dtype == WeightDtype.FLOAT_16


def test_sdxl_drops_flux_sections():
    config = TrainConfig.model_validate({"model_type": "sdxl", "flux1": {"transformer_path": "x"}})
    assert config.flux1 is None


def test_flux1_defaults_and_part_dtypes():
    config = TrainConfig.model_validate({"model_type": "flux1"})
    assert config.flux1 is not None
    assert config.flux2_klein is None
    assert config.flux1.variant == Flux1Variant.DEV
    assert config.flux1.quantization == FluxQuantization.FP8
    assert config.mixed_precision == WeightDtype.BFLOAT_16
    for part in (config.unet, config.text_encoder_1, config.text_encoder_2):
        assert part.weight_dtype == WeightDtype.BFLOAT_16
    assert config.unet.learning_rate == pytest.approx(1e-4)


def test_flux_explicit_values_are_kept():
    config = TrainConfig.model_validate(
        {
            "model_type": "flux1",
            "mixed_precision": "float16",
            "unet": {"weight_dtype": "float16", "learning_rate": 3e-5},
        }
    )
    assert config.mixed_precision == WeightDtype.FLOAT_16
    assert config.unet.weight_dtype == WeightDtype.FLOAT_16
    assert config.unet.learning_rate == pytest.approx(3e-5)


def test_klein_section_and_no_second_encoder_training():
    config = TrainConfig.model_validate(
        {"model_type": "flux2_klein", "text_encoder_2": {"train": True}}
    )
    assert config.flux2_klein is not None
    assert config.flux1 is None
    assert config.text_encoder_2.train is False


def test_text_encoder_training_disables_text_cache():
    config = TrainConfig.model_validate(
        {"model_type": "flux1", "text_encoder_1": {"train": True}, "cache_text_encoder_outputs": True}
    )
    assert config.cache_text_encoder_outputs is False


def test_yaml_roundtrip_keeps_only_active_section():
    config = TrainConfig.model_validate(
        {
            "model_type": "flux2_klein",
            "flux2_klein": {"transformer_path": "/m/klein.safetensors", "quantization": "nf4"},
        }
    )
    data = yaml.safe_load(config.to_yaml())
    assert data["model_type"] == "flux2_klein"
    assert "flux1" not in data
    assert data["flux2_klein"]["transformer_path"] == "/m/klein.safetensors"
    again = TrainConfig.from_yaml(config.to_yaml())
    assert again.model_type == ModelType.FLUX2_KLEIN
    assert again.flux2_klein is not None
    assert again.flux2_klein.quantization == FluxQuantization.NF4


def test_display_base_model_name():
    sdxl = TrainConfig.model_validate({"base_model_name": "a.safetensors"})
    assert sdxl.display_base_model_name() == "a.safetensors"
    flux = TrainConfig.model_validate(
        {"model_type": "flux1", "flux1": {"transformer_path": "/m/flux.safetensors"}}
    )
    assert flux.display_base_model_name() == "/m/flux.safetensors"


def test_gpu_validation_requires_bf16_for_flux():
    with pytest.raises(ValueError, match="bfloat16"):
        validate_gpu_config(
            attention_mechanism="sdpa",
            mixed_precision=WeightDtype.FLOAT_16,
            vae_dtype=VaeDtype.AUTO,
            model_type=ModelType.FLUX1,
        )
    validate_gpu_config(
        attention_mechanism="sdpa",
        mixed_precision=WeightDtype.FLOAT_16,
        vae_dtype=VaeDtype.AUTO,
        model_type=ModelType.SDXL,
    )
