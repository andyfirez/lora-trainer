"""Queues router: list queue, move to top."""

from typing import Sequence

from fastapi import APIRouter

from src.api.dependencies import QueuesServiceDep
from src.api.schemas.queues import QueueEntryResponse, QueueEntryWithJobResponse

router = APIRouter(prefix="/queues", tags=["queues"])


@router.get("/", response_model=list[QueueEntryWithJobResponse])
async def list_queue(service: QueuesServiceDep) -> list[QueueEntryWithJobResponse]:
    pairs = await service.list_queue_with_jobs()
    return [
        QueueEntryWithJobResponse(entry=entry, job=job)  # type: ignore[arg-type]
        for entry, job in pairs
    ]


@router.post("/{job_id}/move-to-top", response_model=QueueEntryResponse)
async def move_to_top(job_id: int, service: QueuesServiceDep) -> QueueEntryResponse:
    entry = await service.move_to_top(job_id)
    return entry  # type: ignore[return-value]
