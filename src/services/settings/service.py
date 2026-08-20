"""Apply and persist runtime application settings."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src import __version__
from src.api.schemas.settings import SettingsPatch, SettingsResponse, TrainingSystemInfo
from src.services.settings.exceptions import (
    EmptySettingsPatchError,
    InvalidGpuDefaultsError,
)
from src.settings.app_settings import settings
from src.settings.config_persist import (
    apply_gpu_defaults,
    apply_storage_settings,
    apply_training_settings,
    get_config_path,
    persist_gpu_defaults,
    persist_storage_settings,
    persist_training_settings,
)
from src.settings.gpu_info import get_gpu_info
from src.trainer.gpu_config_validation import validate_gpu_config

_TRAINING_FIELDS = ("max_concurrent_jobs", "worker_poll_interval_seconds")
_STORAGE_FIELDS = ("datasets_root", "base_models_root", "lora_root")
_GPU_FIELDS = ("tf32", "attention_mechanism", "mixed_precision", "vae_dtype", "sample_vae_tiling")


def _section_kwargs(patch: SettingsPatch, fields: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(patch, name) for name in fields}


def _gpu_persist_kwargs(patch: SettingsPatch) -> dict[str, Any]:
    kwargs = _section_kwargs(patch, _GPU_FIELDS)
    for enum_field in ("mixed_precision", "vae_dtype"):
        value = kwargs[enum_field]
        if value is not None:
            kwargs[enum_field] = value.value
    return kwargs


def _validate_gpu_defaults(patch: SettingsPatch) -> None:
    candidate = settings.gpu_defaults.model_copy(
        update={
            key: value
            for key, value in {
                "tf32": patch.tf32,
                "attention_mechanism": patch.attention_mechanism,
                "mixed_precision": patch.mixed_precision,
                "vae_dtype": patch.vae_dtype,
                "sample_vae_tiling": patch.sample_vae_tiling,
            }.items()
            if value is not None
        }
    )
    try:
        validate_gpu_config(
            attention_mechanism=candidate.attention_mechanism,
            mixed_precision=candidate.mixed_precision,
            vae_dtype=candidate.vae_dtype,
        )
    except ValueError as exc:
        raise InvalidGpuDefaultsError(str(exc)) from exc


@dataclass(frozen=True)
class _SettingsSection:
    fields: tuple[str, ...]
    persist: Callable[..., None]
    apply: Callable[..., object]
    persist_kwargs: Callable[[SettingsPatch], Mapping[str, Any]] | None = None
    before: Callable[[SettingsPatch], None] | None = None


def _sections() -> tuple[_SettingsSection, ...]:
    return (
        _SettingsSection(_TRAINING_FIELDS, persist_training_settings, apply_training_settings),
        _SettingsSection(_STORAGE_FIELDS, persist_storage_settings, apply_storage_settings),
        _SettingsSection(
            _GPU_FIELDS,
            persist_gpu_defaults,
            apply_gpu_defaults,
            persist_kwargs=_gpu_persist_kwargs,
            before=_validate_gpu_defaults,
        ),
    )


class SettingsService:
    def get_settings(self) -> SettingsResponse:
        return SettingsResponse(
            max_concurrent_jobs=settings.training.max_concurrent_jobs,
            worker_poll_interval_seconds=settings.training.worker_poll_interval_seconds,
            server=settings.server,
            database=settings.database,
            storage=settings.storage,
            training=TrainingSystemInfo(
                logs_dir=settings.training.logs_dir,
                cancel_poll_interval_seconds=settings.training.cancel_poll_interval_seconds,
            ),
            gpu_defaults=settings.gpu_defaults,
            config_file=str(get_config_path().resolve()),
            app_version=__version__,
            gpu=get_gpu_info(),
        )

    def apply_patch(self, patch: SettingsPatch) -> SettingsResponse:
        if all(getattr(patch, field) is None for field in SettingsPatch.model_fields):
            raise EmptySettingsPatchError()

        for section in _sections():
            if all(getattr(patch, field) is None for field in section.fields):
                continue
            if section.before is not None:
                section.before(patch)
            kwargs = (
                dict(section.persist_kwargs(patch))
                if section.persist_kwargs is not None
                else _section_kwargs(patch, section.fields)
            )
            section.persist(**kwargs)
            section.apply(**_section_kwargs(patch, section.fields))

        return self.get_settings()
