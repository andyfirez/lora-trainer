"""Sampling SQLModel table — source of truth for sampling config + runtime + results."""

from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from src.db.tables.runnable_mixin import RunnableMixin
from src.db.tables.timestamp_mixin import TimestampMixin


class Sampling(RunnableMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "samplings"

    id: Optional[int] = Field(default=None, primary_key=True)

    config: dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False),
        description="Snapshot JSON — source of truth",
    )
    lora_paths: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=True),
        description="Resolved LoRA file paths",
    )
    progress_step: Optional[int] = Field(default=None)
    progress_total: Optional[int] = Field(default=None)
    progress_status: Optional[str] = Field(default=None)
