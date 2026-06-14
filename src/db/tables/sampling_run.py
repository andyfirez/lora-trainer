"""SamplingRun SQLModel table."""

from enum import StrEnum
from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class SamplingRunStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SamplingRun(TimestampMixin, SQLModel, table=True):
    __tablename__ = "sampling_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    config_yaml: str = Field(description="YAML-serialized TrainConfig used for sampling")
    lora_paths_yaml: str = Field(description="YAML-serialized list of LoRA paths")
    status: SamplingRunStatus = Field(default=SamplingRunStatus.PENDING, index=True)
    source_job_id: Optional[int] = Field(default=None, foreign_key="training_jobs.id", index=True)
    output_path: Optional[str] = Field(default=None)
    log_path: Optional[str] = Field(default=None)
    pid: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    progress_status: Optional[str] = Field(default=None)
    progress_step: Optional[int] = Field(default=None)
    progress_total: Optional[int] = Field(default=None)
