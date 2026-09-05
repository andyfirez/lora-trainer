"""Wake the runnable worker when the FIFO changes."""

import asyncio

from sqlalchemy import event
from sqlmodel.ext.asyncio.session import AsyncSession

_wake: asyncio.Event | None = None


def wake_event() -> asyncio.Event:
    """Return the loop-local wake event, creating it on first use."""
    global _wake
    if _wake is None:
        _wake = asyncio.Event()
    return _wake


def nudge() -> None:
    """Wake the worker poll loop if it is waiting."""
    wake_event().set()


def schedule_nudge(session: AsyncSession) -> None:
    """Wake the worker after this session commits so the queued row is visible."""
    sync = session.sync_session
    if sync.info.get("runnable_nudge_scheduled"):
        return
    sync.info["runnable_nudge_scheduled"] = True

    @event.listens_for(sync, "after_commit", once=True)
    def _on_commit(_session: object) -> None:
        nudge()


def reset_for_tests() -> None:
    """Drop the cached event so the next call binds to the current loop."""
    global _wake
    _wake = None
