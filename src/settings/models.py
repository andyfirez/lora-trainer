"""Nested configuration models for TOML + env settings."""

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class DatabaseSettings(BaseModel):
    path: str = "lora_trainer.db"
    echo: bool = False


class TrainingSettings(BaseModel):
    worker_poll_interval_seconds: int = Field(default=5, ge=1)
    max_concurrent_jobs: int = Field(default=1, ge=1)
    logs_dir: str = "logs"
    cancel_poll_interval_seconds: int = Field(default=1, ge=1)
