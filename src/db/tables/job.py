"""Unified Job SQLModel table for training and sampling."""

from enum import StrEnum
from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class JobType(StrEnum):
    TRAINING = "training"
    SAMPLING = "sampling"
    TAGGING = "tagging"


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(TimestampMixin, SQLModel, table=True):
    __tablename__ = "jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_type: JobType = Field(index=True)
    name: str = Field(index=True)
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    config_id: Optional[int] = Field(default=None, foreign_key="job_configs.id", index=True)
    config_yaml: str = Field(description="YAML-serialized job config")
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
    last_checkpoint_path: Optional[str] = Field(default=None)
    last_checkpoint_epoch: Optional[int] = Field(default=None)
    last_checkpoint_step: Optional[int] = Field(default=None)
    resume_checkpoint_path: Optional[str] = Field(default=None)
    resume_from_epoch: Optional[int] = Field(default=None)
    resume_from_step: Optional[int] = Field(default=None)
    save_checkpoint_requested: bool = Field(default=False)
    lora_paths_yaml: Optional[str] = Field(default=None, description="YAML-serialized list of LoRA paths")
    source_job_id: Optional[int] = Field(default=None, foreign_key="jobs.id", index=True)
    progress_status: Optional[str] = Field(default=None)
