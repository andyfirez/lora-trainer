"""Jobs router: CRUD + enqueue/cancel actions."""

from typing import Sequence

from fastapi import APIRouter

from src.api.dependencies import JobsServiceDep
from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.job_loss import JobLossResponse
from src.api.schemas.jobs import JobCreate, JobResponse, JobUpdate
from src.db.tables.training_job import JobStatus, TrainingJob

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _to_job_response(job: TrainingJob) -> JobResponse:
    payload = JobResponse.model_validate(job, from_attributes=True).model_dump()
    payload["can_resume"] = bool(
        job.status in (JobStatus.FAILED, JobStatus.CANCELLED) and job.last_checkpoint_path
    )
    return JobResponse.model_validate(payload)


@router.get("/", response_model=list[JobResponse])
async def list_jobs(service: JobsServiceDep) -> Sequence[JobResponse]:
    jobs = await service.list_jobs()
    return [_to_job_response(job) for job in jobs]


@router.post("/", response_model=JobResponse, status_code=201)
async def create_job(body: JobCreate, service: JobsServiceDep) -> JobResponse:
    job = await service.create_job(name=body.name, config_yaml=body.config_yaml)
    return _to_job_response(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    return _to_job_response(await service.get_job(job_id))


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(job_id: int, body: JobUpdate, service: JobsServiceDep) -> JobResponse:
    return _to_job_response(await service.update_job(job_id, name=body.name, config_yaml=body.config_yaml))


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: int, service: JobsServiceDep) -> None:
    await service.delete_job(job_id)


@router.post("/{job_id}/enqueue", response_model=JobResponse, status_code=200)
async def enqueue_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    await service.enqueue_job(job_id)
    return _to_job_response(await service.get_job(job_id))


@router.post("/{job_id}/resume", response_model=JobResponse, status_code=200)
async def resume_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    await service.resume_job(job_id)
    return _to_job_response(await service.get_job(job_id))


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: int, service: JobsServiceDep, save_checkpoint: bool = False) -> JobResponse:
    return _to_job_response(await service.cancel_job(job_id, save_checkpoint=save_checkpoint))


@router.get("/{job_id}/logs", response_model=JobLogsResponse)
async def get_job_logs(job_id: int, service: JobsServiceDep, tail: int = 500) -> JobLogsResponse:
    lines = await service.get_job_logs(job_id, tail=tail)
    return JobLogsResponse(lines=lines)


@router.get("/{job_id}/loss", response_model=JobLossResponse)
async def get_job_loss(
    job_id: int,
    service: JobsServiceDep,
    key: str = "loss/loss",
    limit: int = 2000,
    since_step: int | None = None,
    stride: int = 1,
) -> JobLossResponse:
    return await service.get_job_loss(
        job_id,
        key=key,
        limit=limit,
        since_step=since_step,
        stride=stride,
    )
