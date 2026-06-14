"""Configs router: CRUD + create job from config."""

from typing import Sequence

from fastapi import APIRouter, Query

from src.api.converters import to_job_response
from src.api.dependencies import JobConfigServiceDep, JobsServiceDep
from src.api.schemas.configs import (
    CreateJobFromConfigRequest,
    JobConfigCreate,
    JobConfigResponse,
    JobConfigUpdate,
)
from src.api.schemas.jobs import JobResponse
from src.db.tables.job_config import ConfigType

router = APIRouter(prefix="/configs", tags=["configs"])


@router.get("/", response_model=list[JobConfigResponse])
async def list_configs(
    service: JobConfigServiceDep,
    config_type: ConfigType | None = Query(default=None),
) -> Sequence[JobConfigResponse]:
    configs = await service.list_configs(config_type=config_type)
    return [JobConfigResponse.model_validate(c, from_attributes=True) for c in configs]


@router.post("/", response_model=JobConfigResponse, status_code=201)
async def create_config(body: JobConfigCreate, service: JobConfigServiceDep) -> JobConfigResponse:
    config = await service.create_config(
        name=body.name,
        config_type=body.config_type,
        config_yaml=body.config_yaml,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.get("/{config_id}", response_model=JobConfigResponse)
async def get_config(config_id: int, service: JobConfigServiceDep) -> JobConfigResponse:
    config = await service.get_config(config_id)
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.patch("/{config_id}", response_model=JobConfigResponse)
async def update_config(
    config_id: int,
    body: JobConfigUpdate,
    service: JobConfigServiceDep,
) -> JobConfigResponse:
    config = await service.update_config(
        config_id,
        name=body.name,
        config_yaml=body.config_yaml,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.delete("/{config_id}", status_code=204)
async def delete_config(config_id: int, service: JobConfigServiceDep) -> None:
    await service.delete_config(config_id)


@router.post("/{config_id}/jobs", response_model=JobResponse, status_code=201)
async def create_job_from_config(
    config_id: int,
    body: CreateJobFromConfigRequest,
    jobs_service: JobsServiceDep,
) -> JobResponse:
    job = await jobs_service.create_from_config(
        config_id,
        name=body.name,
        lora_paths=body.lora_paths,
        source_job_id=body.source_job_id,
    )
    if body.enqueue and job.id is not None:
        await jobs_service.enqueue_job(job.id)
        job = await jobs_service.get_job(job.id)
    return to_job_response(job, jobs_service)
