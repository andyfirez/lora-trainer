"""Nested configuration models for TOML + env settings."""

from typing import Literal

from pydantic import BaseModel, Field

from src.trainer.config import VaeDtype, WeightDtype


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class DatabaseSettings(BaseModel):
    path: str = "lora_trainer.db"
    echo: bool = False


class TrainingSettings(BaseModel):
    worker_poll_interval_seconds: int = Field(default=5, ge=1)
    max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        description="Max parallel subprocess runnables (loras + samplings). Does not limit tagging.",
    )
    logs_dir: str = "logs"
    cancel_poll_interval_seconds: int = Field(default=1, ge=1)


class StorageSettings(BaseModel):
    datasets_root: str = "~/lora-trainer/datasets"
    base_models_root: str = "~/lora-trainer/base-models"
    lora_root: str = "~/lora-trainer/lora"


class GpuDefaultsSettings(BaseModel):
    tf32: bool = True
    attention_mechanism: Literal["default", "sdpa", "xformers"] = "sdpa"
    mixed_precision: WeightDtype = WeightDtype.FLOAT_16
    vae_dtype: VaeDtype = VaeDtype.AUTO
    sample_vae_tiling: bool = True
