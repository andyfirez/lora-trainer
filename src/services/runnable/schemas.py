"""Domain schemas for runnable job metrics (shared by services; API re-exports)."""

from typing import Optional

from pydantic import BaseModel


class LossPoint(BaseModel):
    step: int
    wall_time: Optional[float] = None
    value: Optional[float] = None


class JobLossResponse(BaseModel):
    key: str
    keys: list[str]
    points: list[LossPoint]
