from unittest.mock import MagicMock

import pytest
from src.api.schemas.settings import SettingsPatch
from src.services.settings.exceptions import EmptySettingsPatchError
from src.services.settings.service import SettingsService


def test_apply_patch_requires_at_least_one_field() -> None:
    with pytest.raises(EmptySettingsPatchError):
        SettingsService().apply_patch(SettingsPatch())


def test_apply_patch_runs_matching_section(monkeypatch) -> None:
    persist_training = MagicMock()
    apply_training = MagicMock()
    persist_storage = MagicMock()
    apply_storage = MagicMock()
    persist_gpu = MagicMock()
    apply_gpu = MagicMock()
    monkeypatch.setattr("src.services.settings.service.persist_training_settings", persist_training)
    monkeypatch.setattr("src.services.settings.service.apply_training_settings", apply_training)
    monkeypatch.setattr("src.services.settings.service.persist_storage_settings", persist_storage)
    monkeypatch.setattr("src.services.settings.service.apply_storage_settings", apply_storage)
    monkeypatch.setattr("src.services.settings.service.persist_gpu_defaults", persist_gpu)
    monkeypatch.setattr("src.services.settings.service.apply_gpu_defaults", apply_gpu)
    monkeypatch.setattr(
        "src.services.settings.service.SettingsService.get_settings",
        lambda self: "ok",
    )

    result = SettingsService().apply_patch(SettingsPatch(max_concurrent_jobs=3))

    assert result == "ok"
    persist_training.assert_called_once_with(
        max_concurrent_jobs=3,
        worker_poll_interval_seconds=None,
    )
    apply_training.assert_called_once_with(
        max_concurrent_jobs=3,
        worker_poll_interval_seconds=None,
    )
    persist_storage.assert_not_called()
    apply_storage.assert_not_called()
    persist_gpu.assert_not_called()
    apply_gpu.assert_not_called()
