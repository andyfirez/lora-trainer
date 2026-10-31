from pathlib import Path

import pytest

from src.storage.base_model_paths import resolve_config_base_model


def test_resolve_checkpoint_under_base_models_root(storage_roots) -> None:
    path = resolve_config_base_model("test-model.safetensors")
    assert path == str((storage_roots["base_models"] / "test-model.safetensors").resolve())


def test_resolve_absolute_checkpoint(tmp_path: Path, storage_roots) -> None:
    checkpoint = tmp_path / "sdxl.safetensors"
    checkpoint.write_bytes(b"")
    assert resolve_config_base_model(str(checkpoint)) == str(checkpoint.resolve())


def test_rejects_huggingface_style_id(storage_roots) -> None:
    with pytest.raises(ValueError, match="not found|checkpoint"):
        resolve_config_base_model("stabilityai/stable-diffusion-xl-base-1.0")


def test_rejects_directory_path(storage_roots) -> None:
    folder = storage_roots["base_models"] / "folder-only"
    folder.mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_config_base_model(str(folder))
