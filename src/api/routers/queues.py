"""Queues router: list queue, move to top."""

from typing import Sequence

import yaml
from fastapi import APIRouter

from src.api.dependencies import QueuesServiceDep
from src.api.schemas.jobs import JobResponse
from src.api.schemas.queues import QueueEntryResponse, QueueEntryWithItemResponse
from src.api.schemas.sampling_runs import SamplingRunResponse
from src.db.tables.queue_entry import QueueItemType
from src.db.tables.sampling_run import SamplingRun
from src.db.tables.training_job import TrainingJob

router = APIRouter(prefix="/queues", tags=["queues"])


def _to_sampling_run_response(sampling_run: SamplingRun) -> SamplingRunResponse:
    payload = sampling_run.model_dump()
    payload["lora_paths"] = yaml.safe_load(sampling_run.lora_paths_yaml) or []
    return SamplingRunResponse.model_validate(payload)


@router.get("/", response_model=list[QueueEntryWithItemResponse])
async def list_queue(service: QueuesServiceDep) -> list[QueueEntryWithItemResponse]:
    pairs = await service.list_queue_with_items()
    responses: list[QueueEntryWithItemResponse] = []
    for entry, item in pairs:
        if isinstance(item, TrainingJob):
            responses.append(
                QueueEntryWithItemResponse(
                    entry=entry,  # type: ignore[arg-type]
                    job=JobResponse.model_validate(item, from_attributes=True),
                )
            )
        elif isinstance(item, SamplingRun):
            responses.append(
                QueueEntryWithItemResponse(
                    entry=entry,  # type: ignore[arg-type]
                    sampling_run=_to_sampling_run_response(item),
                )
            )
    return responses


@router.post("/{item_type}/{item_id}/move-to-top", response_model=QueueEntryResponse)
async def move_to_top(
    item_type: QueueItemType,
    item_id: int,
    service: QueuesServiceDep,
) -> QueueEntryResponse:
    entry = await service.move_to_top(item_type, item_id)
    return entry  # type: ignore[return-value]
