"""Tests for storage browse service."""

from src.services.storage.browse import StorageBrowseService


def test_discover_base_models_lists_checkpoints_and_model_folders(storage_roots) -> None:
    base_models = storage_roots["base_models"]
    (base_models / "flat.safetensors").write_bytes(b"x")
    nested = base_models / "vendor" / "sdxl-base"
    nested.mkdir(parents=True)
    (nested / "model_index.json").write_text("{}", encoding="utf-8")
    (nested / "unet").mkdir()
    (nested / "unet" / "config.json").write_text("{}", encoding="utf-8")

    models = StorageBrowseService().discover_base_models()
    paths = {entry.relative_path for entry in models}
    assert "flat.safetensors" in paths
    assert "vendor/sdxl-base" in paths
    assert "vendor/sdxl-base/unet" not in paths


def test_looks_like_model_dir_detects_diffusers_layout(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert StorageBrowseService._looks_like_model_dir(model_dir) is False

    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    assert StorageBrowseService._looks_like_model_dir(model_dir) is True
