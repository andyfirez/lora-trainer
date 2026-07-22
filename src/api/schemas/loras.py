"""Pydantic schemas for trained LoRA API endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TrainedLoraResponse(BaseModel):
    id: int
    name: str
    job_id: int
    config_id: Optional[int]
    config_yaml: str
    base_model_name: str
    weights_path: str
    work_dir: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TrainedLoraSampleResponse(BaseModel):
    filename: str
    path: str
    url: str
    kind: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class TrainedLoraSamplesResponse(BaseModel):
    samples: list[TrainedLoraSampleResponse] = Field(default_factory=list)


class ReproduceTrainedLoraRequest(BaseModel):
    name: Optional[str] = None
    enqueue: bool = False
