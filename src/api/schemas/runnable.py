"""Shared response fields for Runnable entities (Lora, Sampling)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.db.tables.runnable_mixin import RunnableStatus
from src.services.runnable.runtime import compute_elapsed_seconds


class RunnableResponse(BaseModel):
    id: int
    name: str
    status: RunnableStatus
    queue_position: Optional[int] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    log_path: Optional[str] = None
    running_started_at: Optional[datetime] = None
    accumulated_elapsed_seconds: float
    elapsed_seconds: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def _compute_elapsed_seconds(self) -> "RunnableResponse":
        """Attach a live elapsed-time figure while the entity is running."""
        self.elapsed_seconds = compute_elapsed_seconds(self)  # type: ignore[arg-type]
        return self


class RunnableSampleResponse(BaseModel):
    filename: str
    path: str
    url: str
    kind: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class RunnableSamplesResponse(BaseModel):
    samples: list[RunnableSampleResponse] = Field(default_factory=list)
