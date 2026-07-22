"""Sampling config router: CRUD + create job from sampling config."""

from typing import Sequence

from fastapi import APIRouter

from src.api.converters import to_job_response
from src.api.dependencies import JobConfigServiceDep, JobsServiceDep
from src.api.schemas.configs import (
    CreateJobFromConfigRequest,
    JobConfigCloneRequest,
    JobConfigResponse,
    JobConfigUpdate,
    SamplingConfigCreate,
)
from src.api.schemas.jobs import JobResponse
from src.db.tables.job_config import ConfigType
from src.services.configs.exceptions import JobConfigNotFoundError

router = APIRouter(prefix="/sampling-configs", tags=["sampling-configs"])


async def _get_sampling_config(config_id: int, service: JobConfigServiceDep):
    try:
        config = await service.get_config(config_id)
    except JobConfigNotFoundError:
        raise
    if config.config_type != ConfigType.SAMPLING:
        raise JobConfigNotFoundError(config_id)
    return config


@router.get("/", response_model=list[JobConfigResponse])
async def list_sampling_configs(service: JobConfigServiceDep) -> Sequence[JobConfigResponse]:
    configs = await service.list_configs(config_type=ConfigType.SAMPLING)
    return [JobConfigResponse.model_validate(c, from_attributes=True) for c in configs]


@router.post("/", response_model=JobConfigResponse, status_code=201)
async def create_sampling_config(
    body: SamplingConfigCreate, service: JobConfigServiceDep
) -> JobConfigResponse:
    config = await service.create_config(
        name=body.name,
        config_type=ConfigType.SAMPLING,
        config_yaml=body.config_yaml,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.get("/{config_id}", response_model=JobConfigResponse)
async def get_sampling_config(
    config_id: int, service: JobConfigServiceDep
) -> JobConfigResponse:
    config = await _get_sampling_config(config_id, service)
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.patch("/{config_id}", response_model=JobConfigResponse)
async def update_sampling_config(
    config_id: int,
    body: JobConfigUpdate,
    service: JobConfigServiceDep,
) -> JobConfigResponse:
    await _get_sampling_config(config_id, service)
    config = await service.update_config(
        config_id,
        name=body.name,
        config_yaml=body.config_yaml,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.delete("/{config_id}", status_code=204)
async def delete_sampling_config(config_id: int, service: JobConfigServiceDep) -> None:
    await _get_sampling_config(config_id, service)
    await service.delete_config(config_id)


@router.post("/{config_id}/clone", response_model=JobConfigResponse, status_code=201)
async def clone_sampling_config(
    config_id: int,
    body: JobConfigCloneRequest,
    service: JobConfigServiceDep,
) -> JobConfigResponse:
    await _get_sampling_config(config_id, service)
    config = await service.clone_config(
        config_id,
        name=body.name,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.post("/{config_id}/jobs", response_model=JobResponse, status_code=201)
async def create_job_from_sampling_config(
    config_id: int,
    body: CreateJobFromConfigRequest,
    service: JobConfigServiceDep,
    jobs_service: JobsServiceDep,
) -> JobResponse:
    await _get_sampling_config(config_id, service)
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
