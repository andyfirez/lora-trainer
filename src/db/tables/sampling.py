"""Sampling SQLModel table — source of truth for sampling config + runtime + results."""

from typing import Optional

from sqlmodel import Field, SQLModel

from src.db.tables.runnable_mixin import RunnableMixin
from src.db.tables.timestamp_mixin import TimestampMixin


class Sampling(RunnableMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "samplings"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Sampling always has a config — no "discovered" concept here.
    config_yaml: str = Field(description="Snapshot YAML — source of truth")

    lora_paths_yaml: Optional[str] = Field(
        default=None, description="YAML-serialized list of resolved LoRA paths"
    )
    progress_step: Optional[int] = Field(default=None)
    progress_total: Optional[int] = Field(default=None)
    progress_status: Optional[str] = Field(default=None)
