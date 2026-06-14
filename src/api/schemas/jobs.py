"""Pydantic schemas for training job API endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.db.tables.training_job import JobStatus


class JobCreate(BaseModel):
    name: str
    config_yaml: str


class JobUpdate(BaseModel):
    name: Optional[str] = None
    config_yaml: Optional[str] = None


class JobResponse(BaseModel):
    id: int
    name: str
    config_yaml: str
    status: JobStatus
    output_path: Optional[str]
    log_path: Optional[str]
    pid: Optional[int]
    error_message: Optional[str]
    progress_step: Optional[int]
    progress_total: Optional[int]
    progress_loss: Optional[float]
    progress_avr_loss: Optional[float]
    progress_epoch: Optional[int]
    progress_epoch_total: Optional[int]
    cache_progress_step: Optional[int]
    cache_progress_total: Optional[int]
    sampling_status: Optional[str]
    sampling_step: Optional[int]
    sampling_total: Optional[int]
    last_checkpoint_path: Optional[str]
    last_checkpoint_epoch: Optional[int]
    last_checkpoint_step: Optional[int]
    resume_checkpoint_path: Optional[str]
    resume_from_epoch: Optional[int]
    resume_from_step: Optional[int]
    save_checkpoint_requested: bool
    can_resume: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
