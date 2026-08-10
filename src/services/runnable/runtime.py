"""Shared runtime state transitions for Runnable entities (Lora, Sampling)."""

from datetime import datetime, timezone
from typing import Optional

from src.db.tables.runnable_mixin import RunnableMixin, RunnableStatus


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _accumulate_elapsed(entity: RunnableMixin, *, now: Optional[datetime] = None) -> None:
    if entity.running_started_at is None:
        return
    started_at = _ensure_utc(entity.running_started_at)
    current = now or datetime.now(timezone.utc)
    entity.accumulated_elapsed_seconds = (entity.accumulated_elapsed_seconds or 0.0) + (
        current - started_at
    ).total_seconds()
    entity.running_started_at = None


def remove_from_queue(entity: RunnableMixin) -> None:
    entity.queue_position = None


def mark_running(entity: RunnableMixin, *, pid: int) -> None:
    entity.status = RunnableStatus.RUNNING
    entity.running_started_at = datetime.now(timezone.utc)
    entity.pid = pid


def mark_finished(
    entity: RunnableMixin,
    status: RunnableStatus,
    *,
    error_message: Optional[str] = None,
) -> None:
    """Transition a running entity to a terminal status, accumulating elapsed time."""
    if entity.status == RunnableStatus.RUNNING:
        _accumulate_elapsed(entity)
    entity.status = status
    entity.pid = None
    if error_message is not None:
        entity.error_message = error_message
    elif status == RunnableStatus.COMPLETED:
        entity.error_message = None


def cancel(entity: RunnableMixin) -> None:
    """Cancel a queued or running entity. A running subprocess must still be killed by the caller."""
    if entity.status == RunnableStatus.RUNNING:
        _accumulate_elapsed(entity)
    entity.status = RunnableStatus.CANCELLED
    entity.queue_position = None


def clear_runtime(entity: RunnableMixin) -> None:
    """Reset transient fields shared by all Runnable entities before a (re-)enqueue."""
    entity.pid = None
    entity.error_message = None


def compute_elapsed_seconds(entity: RunnableMixin, *, now: Optional[datetime] = None) -> Optional[float]:
    accumulated = entity.accumulated_elapsed_seconds or 0.0
    has_running_session = entity.running_started_at is not None
    if entity.status == RunnableStatus.RUNNING and has_running_session:
        current = now or datetime.now(timezone.utc)
        started_at = _ensure_utc(entity.running_started_at)  # type: ignore[arg-type]
        return accumulated + (current - started_at).total_seconds()
    if accumulated > 0 or has_running_session:
        return accumulated
    return None
