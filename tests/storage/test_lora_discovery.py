"""Tests for LoRA filesystem discovery."""

from pathlib import Path

from src.services.loras.discovery import LoraDiscoveryService
from src.settings.app_settings import settings


def test_discover_lora_work_dirs_finds_final_and_checkpoint_only(storage_roots) -> None:
    final_dir = storage_roots["lora"] / "final_lora"
    final_dir.mkdir()
    (final_dir / "final_lora.safetensors").write_bytes(b"final")

    checkpoint_dir = storage_roots["lora"] / "nested" / "checkpoint_only"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_only_epoch2.safetensors").write_bytes(b"epoch")

    discovered = LoraDiscoveryService().discover_lora_work_dirs()
    paths = {item.relative_path for item in discovered}
    assert "final_lora" in paths
    assert "nested/checkpoint_only" in paths


def test_discover_lora_work_dirs_uses_configured_root(tmp_path, monkeypatch) -> None:
    lora_root = tmp_path / "custom-lora"
    lora_root.mkdir()
    work_dir = lora_root / "external"
    work_dir.mkdir()
    (work_dir / "external.safetensors").write_bytes(b"x")

    settings.storage = settings.storage.model_copy(update={"lora_root": str(lora_root)})
    discovered = LoraDiscoveryService().discover_lora_work_dirs()
    assert len(discovered) == 1
    assert discovered[0].relative_path == "external"
