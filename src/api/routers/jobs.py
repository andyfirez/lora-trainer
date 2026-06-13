"""Jobs router: CRUD + enqueue/cancel actions."""

from typing import Sequence

from fastapi import APIRouter

from src.api.dependencies import JobsServiceDep
from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.jobs import JobCreate, JobResponse, JobUpdate

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=list[JobResponse])
async def list_jobs(service: JobsServiceDep) -> Sequence[JobResponse]:
    return await service.list_jobs()  # type: ignore[return-value]


@router.post("/", response_model=JobResponse, status_code=201)
async def create_job(body: JobCreate, service: JobsServiceDep) -> JobResponse:
    job = await service.create_job(name=body.name, config_yaml=body.config_yaml)
    return job  # type: ignore[return-value]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    return await service.get_job(job_id)  # type: ignore[return-value]


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(job_id: int, body: JobUpdate, service: JobsServiceDep) -> JobResponse:
    return await service.update_job(job_id, name=body.name, config_yaml=body.config_yaml)  # type: ignore[return-value]


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: int, service: JobsServiceDep) -> None:
    await service.delete_job(job_id)


@router.post("/{job_id}/enqueue", response_model=JobResponse, status_code=200)
async def enqueue_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    await service.enqueue_job(job_id)
    return await service.get_job(job_id)  # type: ignore[return-value]


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    return await service.cancel_job(job_id)  # type: ignore[return-value]


@router.get("/{job_id}/logs", response_model=JobLogsResponse)
async def get_job_logs(job_id: int, service: JobsServiceDep, tail: int = 500) -> JobLogsResponse:
    lines = await service.get_job_logs(job_id, tail=tail)
    return JobLogsResponse(lines=lines)
