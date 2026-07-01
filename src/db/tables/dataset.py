"""Dataset SQLModel table."""

from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class Dataset(TimestampMixin, SQLModel, table=True):
    __tablename__ = "datasets"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    image_dir: str
    caption_dir: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    target_resolution: Optional[int] = Field(default=None, ge=64, le=2048)
    preprocess_ready: bool = Field(default=False)
    enable_bucket: bool = Field(default=False)
    bucket_reso_steps: int = Field(default=64, ge=8, le=512)
    min_bucket_reso: int = Field(default=512, ge=64, le=2048)
    max_bucket_reso: int = Field(default=2048, ge=64, le=2048)
    bucket_no_upscale: bool = Field(default=True)
