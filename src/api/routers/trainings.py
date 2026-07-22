"""Training configs router: CRUD + create job from training config."""

from typing import Sequence

from fastapi import APIRouter

from src.api.converters import to_job_response
from src.api.dependencies import JobConfigServiceDep, JobsServiceDep
from src.api.schemas.configs import (
    CreateJobFromConfigRequest,
    JobConfigCloneRequest,
    JobConfigResponse,
    JobConfigUpdate,
    TrainingConfigCreate,
)
from src.api.schemas.jobs import JobResponse
from src.db.tables.job_config import ConfigType
from src.services.configs.exceptions import JobConfigNotFoundError

router = APIRouter(prefix="/trainings", tags=["trainings"])


async def _require_training_config(config_id: int, service: JobConfigServiceDep):
    config = await service.get_config(config_id)
    if config.config_type != ConfigType.TRAINING:
        raise JobConfigNotFoundError(config_id)
    return config


@router.get("/", response_model=list[JobConfigResponse])
async def list_trainings(service: JobConfigServiceDep) -> Sequence[JobConfigResponse]:
    configs = await service.list_configs(config_type=ConfigType.TRAINING)
    return [JobConfigResponse.model_validate(c, from_attributes=True) for c in configs]


@router.post("/", response_model=JobConfigResponse, status_code=201)
async def create_training(body: TrainingConfigCreate, service: JobConfigServiceDep) -> JobConfigResponse:
    config = await service.create_config(
        name=body.name,
        config_type=ConfigType.TRAINING,
        config_yaml=body.config_yaml,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.get("/{config_id}", response_model=JobConfigResponse)
async def get_training(config_id: int, service: JobConfigServiceDep) -> JobConfigResponse:
    config = await _require_training_config(config_id, service)
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.patch("/{config_id}", response_model=JobConfigResponse)
async def update_training(
    config_id: int,
    body: JobConfigUpdate,
    service: JobConfigServiceDep,
) -> JobConfigResponse:
    await _require_training_config(config_id, service)
    config = await service.update_config(
        config_id,
        name=body.name,
        config_yaml=body.config_yaml,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.delete("/{config_id}", status_code=204)
async def delete_training(config_id: int, service: JobConfigServiceDep) -> None:
    await _require_training_config(config_id, service)
    await service.delete_config(config_id)


@router.post("/{config_id}/clone", response_model=JobConfigResponse, status_code=201)
async def clone_training(
    config_id: int,
    body: JobConfigCloneRequest,
    service: JobConfigServiceDep,
) -> JobConfigResponse:
    await _require_training_config(config_id, service)
    config = await service.clone_config(
        config_id,
        name=body.name,
        description=body.description,
    )
    return JobConfigResponse.model_validate(config, from_attributes=True)


@router.post("/{config_id}/jobs", response_model=JobResponse, status_code=201)
async def create_job_from_training(
    config_id: int,
    body: CreateJobFromConfigRequest,
    jobs_service: JobsServiceDep,
    config_service: JobConfigServiceDep,
) -> JobResponse:
    await _require_training_config(config_id, config_service)
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
