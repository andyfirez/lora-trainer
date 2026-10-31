import pytest

from src.trainer.config import TrainConfig
from src.trainer.flux.model_files import validate_flux_model_files


def _touch(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _tokenizer_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    return str(path)


def test_flux1_requires_every_component(tmp_path):
    config = TrainConfig.model_validate({"model_type": "flux1"})
    with pytest.raises(ValueError, match="transformer"):
        validate_flux_model_files(config)

    flux1 = {
        "transformer_path": _touch(tmp_path / "t.safetensors"),
        "vae_path": _touch(tmp_path / "vae.safetensors"),
        "clip_l": {
            "weights_path": _touch(tmp_path / "clip.safetensors"),
            "tokenizer_dir": _tokenizer_dir(tmp_path / "clip"),
        },
        "t5": {
            "weights_path": _touch(tmp_path / "t5.safetensors"),
            "tokenizer_dir": _tokenizer_dir(tmp_path / "t5"),
        },
    }
    config = TrainConfig.model_validate({"model_type": "flux1", "flux1": flux1})
    validate_flux_model_files(config)


def test_weights_must_be_single_safetensors_file(tmp_path):
    folder = tmp_path / "diffusers"
    folder.mkdir()
    config = TrainConfig.model_validate(
        {"model_type": "flux2_klein", "flux2_klein": {"transformer_path": str(folder)}}
    )
    with pytest.raises(ValueError, match="not found"):
        validate_flux_model_files(config)

    ckpt = _touch(tmp_path / "t.ckpt")
    config = TrainConfig.model_validate(
        {"model_type": "flux2_klein", "flux2_klein": {"transformer_path": ckpt}}
    )
    with pytest.raises(ValueError, match="safetensors"):
        validate_flux_model_files(config)


def test_klein_text_encoder_needs_tokenizer_dir(tmp_path):
    klein = {
        "transformer_path": _touch(tmp_path / "t.safetensors"),
        "vae_path": _touch(tmp_path / "vae.safetensors"),
        "qwen3": {"weights_path": _touch(tmp_path / "q.safetensors"), "tokenizer_dir": ""},
    }
    config = TrainConfig.model_validate({"model_type": "flux2_klein", "flux2_klein": klein})
    with pytest.raises(ValueError, match="tokenizer directory"):
        validate_flux_model_files(config)

    klein["qwen3"]["tokenizer_dir"] = _tokenizer_dir(tmp_path / "qwen")
    config = TrainConfig.model_validate({"model_type": "flux2_klein", "flux2_klein": klein})
    validate_flux_model_files(config)
