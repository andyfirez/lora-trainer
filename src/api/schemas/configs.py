"""Pydantic schemas for job config API endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.db.tables.job_config import ConfigType


class JobConfigCreate(BaseModel):
    name: str
    config_type: ConfigType
    config_yaml: str
    description: Optional[str] = None


class TrainingConfigCreate(BaseModel):
    name: str
    config_yaml: str
    description: Optional[str] = None


class SamplingConfigCreate(BaseModel):
    name: str
    config_yaml: str
    description: Optional[str] = None


class JobConfigUpdate(BaseModel):
    name: Optional[str] = None
    config_yaml: Optional[str] = None
    description: Optional[str] = None


class JobConfigCloneRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class JobConfigResponse(BaseModel):
    id: int
    name: str
    config_type: ConfigType
    config_yaml: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CreateJobFromConfigRequest(BaseModel):
    name: Optional[str] = None
    lora_paths: Optional[list[str]] = None
    source_job_id: Optional[int] = None
    enqueue: bool = False
