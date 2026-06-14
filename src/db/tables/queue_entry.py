"""QueueEntry SQLModel table."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from sqlalchemy import Column, String
from sqlmodel import Field, SQLModel


class QueueItemType(StrEnum):
    TRAINING = "training"
    SAMPLING = "sampling"


class QueueEntry(SQLModel, table=True):
    __tablename__ = "queue_entries"

    id: Optional[int] = Field(default=None, primary_key=True)
    item_type: QueueItemType = Field(
        default=QueueItemType.TRAINING,
        sa_column=Column(String, index=True),
    )
    item_id: int = Field(index=True)
    position: int = Field(index=True)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
