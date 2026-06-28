"""Per-image crop metadata for dataset preprocessing."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class DatasetImageCrop(TimestampMixin, SQLModel, table=True):
    __tablename__ = "dataset_image_crops"

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="datasets.id", index=True)
    filename: str
    crop_center_x: float = Field(ge=0.0, le=1.0)
    crop_center_y: float = Field(ge=0.0, le=1.0)
    source_mtime: float
    baked_at: Optional[datetime] = Field(default=None)
