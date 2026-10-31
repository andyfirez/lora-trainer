"""Shared fields for entities that run as a queued subprocess (Lora, Sampling)."""

from datetime import datetime
from enum import StrEnum
from typing import Optional

import sqlalchemy as sa
from sqlmodel import Field


class RunnableStatus(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ORPHAN = "orphan"


TERMINAL_RUNNABLE_STATUSES = frozenset(
    {RunnableStatus.FAILED, RunnableStatus.CANCELLED, RunnableStatus.ORPHAN}
)


class RunnableMixin:
    """Fields shared by every queued/running entity (Lora, Sampling).

    Domain-specific fields (paths, progress detail, checkpoints, ...) live on
    the concrete table, not here.
    """

    name: str = Field(index=True)
    status: RunnableStatus = Field(
        default=RunnableStatus.DRAFT,
        index=True,
        sa_type=sa.Enum(
            RunnableStatus,
            values_callable=lambda statuses: [status.value for status in statuses],
            native_enum=False,
        ),
    )
    queue_position: Optional[int] = Field(default=None, index=True)
    error_message: Optional[str] = Field(default=None)
    output_path: Optional[str] = Field(default=None)
    log_path: Optional[str] = Field(default=None)
    pid: Optional[int] = Field(default=None)
    running_started_at: Optional[datetime] = Field(default=None)
    accumulated_elapsed_seconds: float = Field(default=0.0)
