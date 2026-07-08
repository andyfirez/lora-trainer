"""Configs router: CRUD + create job from config."""

from typing import Sequence

from fastapi import APIRouter, Query

from src.api.converters import to_job_response
from src.api.dependencies import JobConfigServiceDep, JobsServiceDep
from src.api.schemas.configs import (
    CreateJobFromConfigRequest,
    JobConfigCloneRequest,
    JobConfigCreate,
    JobConfigResponse,
    JobConfigUpdate,
    JobConfigVersionResponse,
    JobConfigVersionSummary,
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


@router.post("/{config_id}/clone", response_model=JobConfigResponse, status_code=201)
async def clone_config(
    config_id: int,
    body: JobConfigCloneRequest,
    service: JobConfigServiceDep,
) -> JobConfigResponse:
    config = await service.clone_config(
        config_id,
        name=body.name,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


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


@router.get("/{config_id}/versions", response_model=list[JobConfigVersionSummary])
async def list_config_versions(
    config_id: int,
    service: JobConfigServiceDep,
) -> list[JobConfigVersionSummary]:
    summaries = await service.list_versions(config_id)
    return [
        JobConfigVersionSummary(
            version=summary.version,
            created_at=summary.created_at,
            lora_name=summary.lora_name,
        )
        for summary in summaries
    ]


@router.get("/{config_id}/versions/{version}", response_model=JobConfigVersionResponse)
async def get_config_version(
    config_id: int,
    version: int,
    service: JobConfigServiceDep,
) -> JobConfigVersionResponse:
    entry = await service.get_version(config_id, version)
    return JobConfigVersionResponse(
        config_id=entry.config_id,
        version=entry.version,
        config_yaml=entry.config_yaml,
        created_at=entry.created_at,
    )
