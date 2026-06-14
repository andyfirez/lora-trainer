"""QueueEntry SQLModel table."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class QueueEntry(SQLModel, table=True):
    __tablename__ = "queue_entries"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobs.id", index=True)
    position: int = Field(index=True)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
