"""Queues router: list queue, move to top."""


from fastapi import APIRouter

from src.api.converters import to_job_response
from src.api.dependencies import JobsServiceDep, QueuesServiceDep
from src.api.schemas.queues import QueueEntryResponse, QueueEntryWithJobResponse

router = APIRouter(prefix="/queues", tags=["queues"])


@router.get("/", response_model=list[QueueEntryWithJobResponse])
async def list_queue(
    service: QueuesServiceDep,
    jobs_service: JobsServiceDep,
) -> list[QueueEntryWithJobResponse]:
    pairs = await service.list_queue_with_jobs()
    responses: list[QueueEntryWithJobResponse] = []
    for entry, job in pairs:
        responses.append(
            QueueEntryWithJobResponse(
                entry=QueueEntryResponse.model_validate(entry, from_attributes=True),
                job=to_job_response(job, jobs_service),
            )
        )
    return responses


@router.post("/{job_id}/move-to-top", response_model=QueueEntryResponse)
async def move_to_top(job_id: int, service: QueuesServiceDep) -> QueueEntryResponse:
    entry = await service.move_to_top(job_id)
    return QueueEntryResponse.model_validate(entry, from_attributes=True)
