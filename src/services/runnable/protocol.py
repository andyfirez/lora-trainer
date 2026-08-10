"""Structural type shared by Lora and Sampling for queue/worker code."""

from typing import Optional, Protocol

from src.db.tables.runnable_mixin import RunnableStatus


class Queueable(Protocol):
    id: Optional[int]
    status: RunnableStatus
    queue_position: Optional[int]
