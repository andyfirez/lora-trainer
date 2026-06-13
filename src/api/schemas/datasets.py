"""Pydantic schemas for dataset API endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DatasetCreate(BaseModel):
    name: str
    image_dir: str
    caption_dir: Optional[str] = None
    description: Optional[str] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    image_dir: Optional[str] = None
    caption_dir: Optional[str] = None
    description: Optional[str] = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    image_dir: str
    caption_dir: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetImagesResponse(BaseModel):
    dataset_id: int
    image_dir: str
    images: list[str]
