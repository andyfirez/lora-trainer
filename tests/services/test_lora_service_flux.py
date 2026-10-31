"""LoraService behaviour for Flux model types and the model_type column."""

import sqlite3

import pytest
from src.services.loras.service import LoraService
from src.services.runnable.exceptions import RunnableValidationError


def _flux1_yaml(dataset_id: int, root, *, complete: bool = True) -> str:
    def touch(name: str) -> str:
        path = root / name
        path.write_bytes(b"x")
        return path.as_posix()

    def tokenizer(name: str) -> str:
        folder = root / name
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text("{}")
        (folder / "tokenizer.json").write_text("{}")
        return folder.as_posix()

    vae = touch("vae.safetensors") if complete else ""
    return f"""model_type: flux1
output_dir: ""
flux1:
  transformer_path: {touch("flux.safetensors")}
  vae_path: "{vae}"
  clip_l:
    weights_path: {touch("clip.safetensors")}
    tokenizer_dir: {tokenizer("clip")}
  t5:
    weights_path: {touch("t5.safetensors")}
    tokenizer_dir: {tokenizer("t5")}
concepts:
  - dataset_id: {dataset_id}
"""


@pytest.mark.asyncio
async def test_create_flux1_lora_stores_model_type(lora_service: LoraService, training_dataset, storage_roots) -> None:
    yaml_text = _flux1_yaml(training_dataset.id, storage_roots["base_models"])
    lora = await lora_service.create_lora(name="flux-one", config_yaml=yaml_text)
    assert lora.model_type == "flux1"
    assert lora.base_model_name.endswith("flux.safetensors")
    assert "model_type: flux1" in lora.config_yaml


@pytest.mark.asyncio
async def test_create_flux1_lora_rejects_missing_component(
    lora_service: LoraService, training_dataset, storage_roots
) -> None:
    yaml_text = _flux1_yaml(training_dataset.id, storage_roots["base_models"], complete=False)
    with pytest.raises(RunnableValidationError, match="VAE"):
        await lora_service.create_lora(name="flux-bad", config_yaml=yaml_text)


@pytest.mark.asyncio
async def test_sdxl_lora_defaults_to_sdxl_model_type(lora_service: LoraService, minimal_training_yaml: str) -> None:
    lora = await lora_service.create_lora(name="sdxl-one", config_yaml=minimal_training_yaml)
    assert lora.model_type == "sdxl"
    assert lora.base_model_name == "test-model.safetensors"


def test_migration_adds_model_type_column_with_sdxl_default(tmp_path, monkeypatch) -> None:
    from src.db.alembic_runner import run_migrations

    db_path = tmp_path / "migrated.db"
    monkeypatch.setattr("src.settings.app_settings.settings.database.path", str(db_path))
    run_migrations()

    conn = sqlite3.connect(db_path)
    columns = {row[1]: row for row in conn.execute("PRAGMA table_info(loras)").fetchall()}
    conn.execute("INSERT INTO loras (name, status, created_at, updated_at, relative_path, weights_relpath, base_model_name, save_checkpoint_requested, accumulated_elapsed_seconds) VALUES ('old', 'draft', '2020-01-01', '2020-01-01', '', '', 'unknown', 0, 0)")
    row = conn.execute("SELECT model_type FROM loras WHERE name='old'").fetchone()
    conn.close()
    assert "model_type" in columns
    assert row == ("sdxl",)

