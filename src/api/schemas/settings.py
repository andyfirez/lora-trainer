"""Pydantic schemas for application settings API."""

from pydantic import BaseModel, Field

from src.settings.gpu_info import GpuInfo
from src.settings.models import DatabaseSettings, ServerSettings


class TrainingSystemInfo(BaseModel):
    logs_dir: str
    cancel_poll_interval_seconds: int


class SettingsResponse(BaseModel):
    max_concurrent_jobs: int
    worker_poll_interval_seconds: int
    server: ServerSettings
    database: DatabaseSettings
    training: TrainingSystemInfo
    config_file: str
    app_version: str
    gpu: GpuInfo


class SettingsPatch(BaseModel):
    max_concurrent_jobs: int | None = Field(default=None, ge=1)
    worker_poll_interval_seconds: int | None = Field(default=None, ge=1)
