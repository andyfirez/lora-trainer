"""Pydantic schemas for application settings API."""

from typing import Literal

from pydantic import BaseModel, Field

from src.settings.gpu_info import GpuInfo
from src.settings.models import DatabaseSettings, GpuDefaultsSettings, ServerSettings, StorageSettings
from src.trainer.config import VaeDtype, WeightDtype


class TrainingSystemInfo(BaseModel):
    logs_dir: str
    cancel_poll_interval_seconds: int


class SettingsResponse(BaseModel):
    max_concurrent_jobs: int
    worker_poll_interval_seconds: int
    server: ServerSettings
    database: DatabaseSettings
    storage: StorageSettings
    training: TrainingSystemInfo
    gpu_defaults: GpuDefaultsSettings
    config_file: str
    app_version: str
    gpu: GpuInfo


class SettingsPatch(BaseModel):
    max_concurrent_jobs: int | None = Field(default=None, ge=1)
    worker_poll_interval_seconds: int | None = Field(default=None, ge=1)
    datasets_root: str | None = None
    base_models_root: str | None = None
    lora_root: str | None = None
    tf32: bool | None = None
    attention_mechanism: Literal["default", "sdpa", "xformers"] | None = None
    mixed_precision: WeightDtype | None = None
    vae_dtype: VaeDtype | None = None
    sample_vae_tiling: bool | None = None
