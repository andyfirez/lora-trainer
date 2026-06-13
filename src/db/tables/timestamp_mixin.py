"""Timestamp mixin for SQLModel tables."""

from datetime import datetime, timezone

from sqlmodel import Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)
