"""Pydantic schemas for dataset API endpoints."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


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


class DatasetItemResponse(BaseModel):
    filename: str
    tags: list[str]
    has_caption: bool


class DatasetItemsResponse(BaseModel):
    dataset_id: int
    items: list[DatasetItemResponse]


class CaptionResponse(BaseModel):
    filename: str
    tags: list[str]


class CaptionUpdateRequest(BaseModel):
    tags: list[str]


class TagStatResponse(BaseModel):
    tag: str
    count: int


class TagStatsResponse(BaseModel):
    tags: list[TagStatResponse]


class BulkTagRequest(BaseModel):
    tag: str
    filenames: list[str] | None = None
    caption_extension: str = ".txt"


class BulkTagResponse(BaseModel):
    updated_count: int


TaggingMode = Literal["if_empty", "overwrite", "append"]


class AutotagRequest(BaseModel):
    mode: TaggingMode = "if_empty"
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    model: str = "wd-v1-4-convnextv2-tagger-v2"
    caption_extension: str = ".txt"
    strip_rating: bool = True
    filenames: list[str] | None = None
    enqueue: bool = True


class AutotagResponse(BaseModel):
    job_id: int
