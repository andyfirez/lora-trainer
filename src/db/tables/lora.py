"""Lora SQLModel table — source of truth for training config + runtime + artifacts."""

from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.runnable_mixin import RunnableMixin
from src.db.tables.timestamp_mixin import TimestampMixin


class Lora(RunnableMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "loras"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Artifacts (empty until the first run resolves them; required once completed)
    relative_path: str = Field(default="", description="Work directory relative to lora_root")
    weights_relpath: str = Field(default="", description="Weights file relative to lora_root")
    base_model_name: str = Field(default="unknown", index=True)

    # Training progress
    progress_step: Optional[int] = Field(default=None)
    progress_total: Optional[int] = Field(default=None)
    progress_loss: Optional[float] = Field(default=None)
    progress_avr_loss: Optional[float] = Field(default=None)
    progress_epoch: Optional[int] = Field(default=None)
    progress_epoch_total: Optional[int] = Field(default=None)
    cache_progress_step: Optional[int] = Field(default=None)
    cache_progress_total: Optional[int] = Field(default=None)

    # Checkpoints / resume
    last_checkpoint_path: Optional[str] = Field(default=None)
    last_checkpoint_epoch: Optional[int] = Field(default=None)
    last_checkpoint_step: Optional[int] = Field(default=None)
    resume_checkpoint_path: Optional[str] = Field(default=None)
    resume_from_epoch: Optional[int] = Field(default=None)
    resume_from_step: Optional[int] = Field(default=None)
    save_checkpoint_requested: bool = Field(default=False)
