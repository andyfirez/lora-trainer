"""Pydantic schemas for queue API endpoints."""

from datetime import datetime

from pydantic import BaseModel

from src.api.schemas.jobs import JobResponse
from src.api.schemas.sampling_runs import SamplingRunResponse
from src.db.tables.queue_entry import QueueItemType


class QueueEntryResponse(BaseModel):
    id: int
    item_type: QueueItemType
    item_id: int
    position: int
    added_at: datetime

    model_config = {"from_attributes": True}


class QueueEntryWithItemResponse(BaseModel):
    entry: QueueEntryResponse
    job: JobResponse | None = None
    sampling_run: SamplingRunResponse | None = None
