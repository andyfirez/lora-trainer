"""Pydantic schemas for job API endpoints."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.db.tables.job import JobStatus, JobType


class TrainingJobDetails(BaseModel):
    progress_loss: Optional[float] = None
    progress_avr_loss: Optional[float] = None
    progress_epoch: Optional[int] = None
    progress_epoch_total: Optional[int] = None
    cache_progress_step: Optional[int] = None
    cache_progress_total: Optional[int] = None
    sampling_status: Optional[str] = None
    sampling_step: Optional[int] = None
    sampling_total: Optional[int] = None
    last_checkpoint_path: Optional[str] = None
    last_checkpoint_epoch: Optional[int] = None
    last_checkpoint_step: Optional[int] = None
    resume_checkpoint_path: Optional[str] = None
    resume_from_epoch: Optional[int] = None
    resume_from_step: Optional[int] = None
    save_checkpoint_requested: bool = False
    sampling_config_id: Optional[int] = None


class SamplingJobDetails(BaseModel):
    lora_paths: list[str] = []
    source_job_id: Optional[int] = None
    progress_status: Optional[str] = None


class TaggingJobDetails(BaseModel):
    progress_status: Optional[str] = None
    dataset_id: int


SampleKind = Literal["cell", "grid", "legacy"]


class JobSampleResponse(BaseModel):
    filename: str
    path: str
    url: str
    kind: SampleKind = "legacy"
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobSamplesResponse(BaseModel):
    samples: list[JobSampleResponse]


class ManifestImageEntryResponse(BaseModel):
    index: int
    file: str
    url: str
    params: dict[str, Any] = Field(default_factory=dict)
    grid_position: dict[str, int] | None = None


class ManifestGridAxisResponse(BaseModel):
    param: str
    values: list[Any] = Field(default_factory=list)


class ManifestGridEntryResponse(BaseModel):
    index: int
    file: str
    url: str
    slice: dict[str, Any] = Field(default_factory=dict)
    x: ManifestGridAxisResponse
    y: ManifestGridAxisResponse
    cells: list[list[int | None]] = Field(default_factory=list)
    title: str = ""


class SweepManifestResponse(BaseModel):
    version: int = 1
    config_id: int | None = None
    job_id: int | None = None
    total_images: int = 0
    images: list[ManifestImageEntryResponse] = Field(default_factory=list)
    grids: list[ManifestGridEntryResponse] = Field(default_factory=list)


class JobResponse(BaseModel):
    id: int
    job_type: JobType
    name: str
    status: JobStatus
    config_id: Optional[int]
    config_version: Optional[int] = None
    config_yaml: str
    output_path: Optional[str]
    log_path: Optional[str]
    pid: Optional[int]
    error_message: Optional[str]
    progress_step: Optional[int]
    progress_total: Optional[int]
    training: Optional[TrainingJobDetails] = None
    sampling: Optional[SamplingJobDetails] = None
    tagging: Optional[TaggingJobDetails] = None
    can_resume: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
