"""Immutable record of a successfully trained LoRA."""

from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.timestamp_mixin import TimestampMixin


class TrainedLora(TimestampMixin, SQLModel, table=True):
    __tablename__ = "trained_loras"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, description="Display name, typically the runtime lora_name")
    job_id: int = Field(foreign_key="jobs.id", unique=True, index=True)
    config_id: Optional[int] = Field(default=None, foreign_key="job_configs.id", index=True)
    config_yaml: str = Field(description="Frozen training config used for this run")
    base_model_name: str = Field(index=True)
    weights_path: str = Field(description="Path to final LoRA weights file")
    work_dir: str = Field(description="Training output directory for this run")
