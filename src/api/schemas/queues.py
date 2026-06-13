"""Pydantic schemas for queue API endpoints."""

from datetime import datetime

from pydantic import BaseModel

from src.api.schemas.jobs import JobResponse


class QueueEntryResponse(BaseModel):
    id: int
    job_id: int
    position: int
    added_at: datetime

    model_config = {"from_attributes": True}


class QueueEntryWithJobResponse(BaseModel):
    entry: QueueEntryResponse
    job: JobResponse
