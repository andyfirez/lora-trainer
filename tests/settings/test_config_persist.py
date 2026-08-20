"""Tests for config.toml persist helpers."""

from src.settings.config_persist import persist_training_settings


def test_persist_training_settings_fills_missing_toml_sections(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[training]
max_concurrent_jobs = 1
worker_poll_interval_seconds = 5
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_path))

    persist_training_settings(max_concurrent_jobs=2)

    saved = config_path.read_text(encoding="utf-8")
    assert "max_concurrent_jobs = 2" in saved
    assert "[server]" in saved
    assert "[database]" in saved
    assert "[storage]" in saved
    assert "[gpu_defaults]" in saved
