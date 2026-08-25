"""Sampling SQLModel table — source of truth for sampling config + runtime + results."""

from enum import StrEnum
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from src.db.tables.runnable_mixin import RunnableMixin
from src.db.tables.timestamp_mixin import TimestampMixin


class SamplingKind(StrEnum):
    EXPERIMENT = "experiment"
    PLAYGROUND = "playground"


class Sampling(RunnableMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "samplings"

    id: Optional[int] = Field(default=None, primary_key=True)

    kind: SamplingKind = Field(
        default=SamplingKind.PLAYGROUND,
        index=True,
        sa_type=sa.Enum(
            SamplingKind,
            values_callable=lambda kinds: [kind.value for kind in kinds],
            native_enum=False,
        ),
    )
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
