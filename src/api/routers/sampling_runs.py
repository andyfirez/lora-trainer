"""Sampling runs router."""

from pathlib import Path
from typing import Sequence

import yaml
from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api.dependencies import SamplingServiceDep
from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.sampling_runs import (
    SamplingRunCreate,
    SamplingRunResponse,
    SamplingRunSampleResponse,
    SamplingRunSamplesResponse,
)
from src.db.tables.sampling_run import SamplingRun
from src.services.sampling.exceptions import SamplingRunNotFoundError

router = APIRouter(prefix="/sampling-runs", tags=["sampling-runs"])


def _to_sampling_run_response(sampling_run: SamplingRun) -> SamplingRunResponse:
    payload = sampling_run.model_dump()
    payload["lora_paths"] = yaml.safe_load(sampling_run.lora_paths_yaml) or []
    return SamplingRunResponse.model_validate(payload)


@router.get("/", response_model=list[SamplingRunResponse])
async def list_sampling_runs(
    service: SamplingServiceDep,
    source_job_id: int | None = None,
) -> Sequence[SamplingRunResponse]:
    runs = await service.list_runs(source_job_id=source_job_id)
    return [_to_sampling_run_response(run) for run in runs]


@router.post("/", response_model=SamplingRunResponse, status_code=201)
async def create_sampling_run(body: SamplingRunCreate, service: SamplingServiceDep) -> SamplingRunResponse:
    sampling_run = await service.create_run(
        name=body.name,
        config_yaml=body.config_yaml,
        lora_paths=body.lora_paths,
        source_job_id=body.source_job_id,
    )
    if body.enqueue:
        await service.enqueue_run(sampling_run.id)
        sampling_run = await service.get_run(sampling_run.id)
    return _to_sampling_run_response(sampling_run)


@router.get("/{sampling_run_id}", response_model=SamplingRunResponse)
async def get_sampling_run(sampling_run_id: int, service: SamplingServiceDep) -> SamplingRunResponse:
    return _to_sampling_run_response(await service.get_run(sampling_run_id))


@router.post("/{sampling_run_id}/enqueue", response_model=SamplingRunResponse)
async def enqueue_sampling_run(sampling_run_id: int, service: SamplingServiceDep) -> SamplingRunResponse:
    await service.enqueue_run(sampling_run_id)
    return _to_sampling_run_response(await service.get_run(sampling_run_id))


@router.post("/{sampling_run_id}/cancel", response_model=SamplingRunResponse)
async def cancel_sampling_run(sampling_run_id: int, service: SamplingServiceDep) -> SamplingRunResponse:
    return _to_sampling_run_response(await service.cancel_run(sampling_run_id))


@router.get("/{sampling_run_id}/logs", response_model=JobLogsResponse)
async def get_sampling_run_logs(
    sampling_run_id: int,
    service: SamplingServiceDep,
    tail: int = 500,
) -> JobLogsResponse:
    lines = await service.get_run_logs(sampling_run_id, tail=tail)
    return JobLogsResponse(lines=lines)


@router.get("/{sampling_run_id}/samples", response_model=SamplingRunSamplesResponse)
async def list_sampling_samples(
    sampling_run_id: int,
    service: SamplingServiceDep,
) -> SamplingRunSamplesResponse:
    sampling_run = await service.get_run(sampling_run_id)
    samples = [
        SamplingRunSampleResponse(
            filename=path.name,
            path=str(path),
            url=f"/sampling-runs/{sampling_run_id}/samples/{path.name}",
        )
        for path in service.list_samples(sampling_run)
    ]
    return SamplingRunSamplesResponse(samples=samples)


@router.get("/{sampling_run_id}/samples/{filename}")
async def get_sampling_sample(
    sampling_run_id: int,
    filename: str,
    service: SamplingServiceDep,
) -> FileResponse:
    sampling_run = await service.get_run(sampling_run_id)
    for path in service.list_samples(sampling_run):
        if path.name == Path(filename).name:
            return FileResponse(path)
    raise SamplingRunNotFoundError(sampling_run_id)
