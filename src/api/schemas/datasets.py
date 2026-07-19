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
    target_resolution: Optional[int] = Field(default=None, ge=64, le=2048)
    enable_bucket: Optional[bool] = None
    bucket_reso_steps: Optional[int] = Field(default=None, ge=8, le=512)
    min_bucket_reso: Optional[int] = Field(default=None, ge=64, le=2048)
    max_bucket_reso: Optional[int] = Field(default=None, ge=64, le=2048)
    bucket_no_upscale: Optional[bool] = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    image_dir: str
    caption_dir: Optional[str]
    description: Optional[str]
    target_resolution: Optional[int]
    preprocess_ready: bool
    enable_bucket: bool
    bucket_reso_steps: int
    min_bucket_reso: int
    max_bucket_reso: int
    bucket_no_upscale: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PreprocessStatusResponse(BaseModel):
    target_resolution: Optional[int]
    preprocess_ready: bool
    total: int
    no_crop: int
    stale: int
    cropped: int
    ready: int


class CropMetaResponse(BaseModel):
    crop_center_x: float
    crop_center_y: float
    fitted_width: int
    fitted_height: int
    source_width: int
    source_height: int
    state: str
    enable_bucket: bool = False
    bucket_width: Optional[int] = None
    bucket_height: Optional[int] = None
    scale_to_width: Optional[int] = None
    scale_to_height: Optional[int] = None
    crop_x: int = 0
    crop_y: int = 0


class CropUpdateRequest(BaseModel):
    crop_center_x: float = Field(ge=0.0, le=1.0)
    crop_center_y: float = Field(ge=0.0, le=1.0)


class BakeRequest(BaseModel):
    filenames: list[str] | None = None


class BakeResponse(BaseModel):
    baked_count: int
    preprocess_ready: bool


class RemoveDuplicatesResponse(BaseModel):
    removed_count: int


class DuplicatesResponse(BaseModel):
    duplicate_count: int


class DatasetImagesResponse(BaseModel):
    dataset_id: int
    image_dir: str
    images: list[str]


class DatasetItemResponse(BaseModel):
    filename: str
    tags: list[str]
    has_caption: bool
    preprocess_state: Optional[str] = None


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
