"""Shared background asyncio loop for subprocess progress DB updates."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def start_progress_loop(name: str) -> asyncio.AbstractEventLoop:
    """Start a daemon thread running a dedicated asyncio event loop."""
    loop = asyncio.new_event_loop()
    threading.Thread(
        target=loop.run_forever,
        daemon=True,
        name=name,
    ).start()
    return loop


def submit_to_progress_loop(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, None],
) -> None:
    """Schedule a coroutine on the progress loop and log failures."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)

    def _log_exception(fut: asyncio.Future[object]) -> None:
        try:
            fut.result()
        except Exception:
            logger.exception("Progress DB update failed")

    future.add_done_callback(_log_exception)


def run_in_progress_loop(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    *,
    timeout_s: float = 1.0,
) -> Any:
    """Run a coroutine on the progress loop and block for the result."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout_s)
