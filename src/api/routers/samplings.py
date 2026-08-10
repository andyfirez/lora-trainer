"""Sampling router — create/enqueue/cancel and result access."""

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api.dependencies import SamplingServiceDep
from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.samplings import (
    CreateSamplingRequest,
    SamplingResponse,
    SamplingSampleResponse,
    SamplingSamplesResponse,
)
from src.db.tables.sampling import Sampling
from src.sampler.sweep.manifest import SweepManifest
from src.services.sampling.service import SamplingService

router = APIRouter(prefix="/samplings", tags=["samplings"])


def _to_response(sampling: Sampling, service: SamplingService) -> SamplingResponse:
    data = sampling.model_dump()
    data["lora_paths"] = service.get_lora_paths(sampling)
    return SamplingResponse.model_validate(data)


def _sample_url(sampling_id: int, relative: str) -> str:
    return f"/samplings/{sampling_id}/sample-file/{relative.replace(chr(92), '/')}"


@router.get("/", response_model=list[SamplingResponse])
async def list_samplings(service: SamplingServiceDep) -> list[SamplingResponse]:
    samplings = await service.list_samplings()
    return [_to_response(sampling, service) for sampling in samplings]


@router.post("/", response_model=SamplingResponse, status_code=201)
async def create_sampling(body: CreateSamplingRequest, service: SamplingServiceDep) -> SamplingResponse:
    sampling = await service.create_sampling(name=body.name, config_yaml=body.config_yaml, lora_paths=body.lora_paths)
    return _to_response(sampling, service)


@router.get("/{sampling_id}", response_model=SamplingResponse)
async def get_sampling(sampling_id: int, service: SamplingServiceDep) -> SamplingResponse:
    sampling = await service.get_sampling(sampling_id)
    return _to_response(sampling, service)


@router.post("/{sampling_id}/enqueue", response_model=SamplingResponse)
async def enqueue_sampling(sampling_id: int, service: SamplingServiceDep) -> SamplingResponse:
    sampling = await service.enqueue_sampling(sampling_id)
    return _to_response(sampling, service)


@router.post("/{sampling_id}/cancel", response_model=SamplingResponse)
async def cancel_sampling(sampling_id: int, service: SamplingServiceDep) -> SamplingResponse:
    sampling = await service.cancel_sampling(sampling_id)
    return _to_response(sampling, service)


@router.get("/{sampling_id}/logs", response_model=JobLogsResponse)
async def get_sampling_logs(sampling_id: int, service: SamplingServiceDep, tail: int = 500) -> JobLogsResponse:
    lines = await service.get_logs(sampling_id, tail=tail)
    return JobLogsResponse(lines=lines)


@router.get("/{sampling_id}/samples", response_model=SamplingSamplesResponse)
async def get_sampling_samples(sampling_id: int, service: SamplingServiceDep) -> SamplingSamplesResponse:
    sampling = await service.get_sampling(sampling_id)
    output_dir = sampling.output_path
    samples = []
    if output_dir:
        from pathlib import Path

        base = Path(output_dir)
        for sample, kind, metadata in service.list_samples(sampling):
            relative = sample.relative_to(base).as_posix()
            samples.append(
                SamplingSampleResponse(
                    filename=sample.name,
                    path=str(sample),
                    url=_sample_url(sampling_id, relative),
                    kind=kind,  # type: ignore[arg-type]
                    metadata=metadata,
                )
            )
    return SamplingSamplesResponse(samples=samples)


@router.get("/{sampling_id}/sweep-manifest", response_model=Optional[SweepManifest])
async def get_sweep_manifest(sampling_id: int, service: SamplingServiceDep) -> SweepManifest | None:
    sampling = await service.get_sampling(sampling_id)
    return service.get_sweep_manifest(sampling)


@router.get("/{sampling_id}/sample-file/{file_path:path}")
async def get_sampling_sample_file(sampling_id: int, file_path: str, service: SamplingServiceDep) -> FileResponse:
    sampling = await service.get_sampling(sampling_id)
    target = service.sample_file_path(sampling, file_path)
    return FileResponse(target)
