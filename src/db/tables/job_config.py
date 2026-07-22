"""JobConfig SQLModel table."""

from enum import StrEnum
from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class ConfigType(StrEnum):
    TRAINING = "training"
    SAMPLING = "sampling"


class JobConfig(TimestampMixin, SQLModel, table=True):
    __tablename__ = "job_configs"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    config_type: ConfigType = Field(index=True)
    config_yaml: str = Field(description="YAML-serialized training or sampling config")
