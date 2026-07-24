"""Immutable record of a trained or discovered LoRA."""

from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class TrainedLora(TimestampMixin, SQLModel, table=True):
    __tablename__ = "trained_loras"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, description="Display name, typically the work dir basename")
    relative_path: str = Field(description="Work directory path relative to lora_root")
    weights_relpath: str = Field(description="Weights file path relative to lora_root")
    job_id: Optional[int] = Field(default=None, foreign_key="jobs.id", unique=True, index=True)
    config_id: Optional[int] = Field(default=None, foreign_key="job_configs.id", index=True)
    config_yaml: Optional[str] = Field(default=None, description="Frozen training config when linked to a job")
    base_model_name: str = Field(index=True)
