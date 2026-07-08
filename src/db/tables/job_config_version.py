"""Version history for training job configs."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobConfigVersion(SQLModel, table=True):
    __tablename__ = "job_config_versions"

    id: Optional[int] = Field(default=None, primary_key=True)
    config_id: int = Field(foreign_key="job_configs.id", index=True)
    version: int = Field(ge=1)
    config_yaml: str = Field(description="YAML-serialized training config for this version")
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
