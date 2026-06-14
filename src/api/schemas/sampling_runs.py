"""Pydantic schemas for sampling run API endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.db.tables.sampling_run import SamplingRunStatus


class SamplingRunCreate(BaseModel):
    name: Optional[str] = None
    config_yaml: str
    lora_paths: list[str] = Field(min_length=1)
    source_job_id: Optional[int] = None
    enqueue: bool = True


class JobSamplingRunCreate(BaseModel):
    lora_paths: list[str] = Field(min_length=1)
    name: Optional[str] = None
    enqueue: bool = True


class SamplingRunResponse(BaseModel):
    id: int
    name: str
    config_yaml: str
    lora_paths: list[str]
    status: SamplingRunStatus
    source_job_id: Optional[int]
    output_path: Optional[str]
    log_path: Optional[str]
    pid: Optional[int]
    error_message: Optional[str]
    progress_status: Optional[str]
    progress_step: Optional[int]
    progress_total: Optional[int]
    created_at: datetime
    updated_at: datetime


class SamplingRunSampleResponse(BaseModel):
    filename: str
    path: str
    url: str


class SamplingRunSamplesResponse(BaseModel):
    samples: list[SamplingRunSampleResponse]
