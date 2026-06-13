"""TrainingJob SQLModel table."""

from enum import StrEnum
from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(TimestampMixin, SQLModel, table=True):
    __tablename__ = "training_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    config_yaml: str = Field(description="YAML-serialized TrainConfig")
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    output_path: Optional[str] = Field(default=None)
    log_path: Optional[str] = Field(default=None)
    pid: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    progress_step: Optional[int] = Field(default=None)
    progress_total: Optional[int] = Field(default=None)
    progress_loss: Optional[float] = Field(default=None)
    progress_avr_loss: Optional[float] = Field(default=None)
    progress_epoch: Optional[int] = Field(default=None)
    progress_epoch_total: Optional[int] = Field(default=None)
    cache_progress_step: Optional[int] = Field(default=None)
    cache_progress_total: Optional[int] = Field(default=None)
    sampling_status: Optional[str] = Field(default=None)
    sampling_step: Optional[int] = Field(default=None)
    sampling_total: Optional[int] = Field(default=None)
