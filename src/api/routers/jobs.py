"""Jobs router: list/get + enqueue/cancel actions."""

from pathlib import Path
from typing import Sequence

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from src.api.converters import to_job_response
from src.api.dependencies import JobsServiceDep
from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.job_loss import JobLossResponse
from src.api.schemas.jobs import (
    JobResponse,
    JobSampleResponse,
    JobSamplesResponse,
    ManifestGridAxisResponse,
    ManifestGridEntryResponse,
    ManifestImageEntryResponse,
    SweepManifestResponse,
)
from src.db.tables.job import JobType
from src.services.jobs.exceptions import JobOperationNotSupportedError

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _sample_url(job_id: int, relative: str) -> str:
    return f"/jobs/{job_id}/sample-file/{relative.replace(chr(92), '/')}"


@router.get("/", response_model=list[JobResponse])
async def list_jobs(
    service: JobsServiceDep,
    job_type: JobType | None = Query(default=None),
    source_job_id: int | None = Query(default=None),
) -> Sequence[JobResponse]:
    if source_job_id is not None:
        jobs = await service.list_jobs_by_source(source_job_id)
    else:
        jobs = await service.list_jobs(job_type=job_type)
    return [to_job_response(job, service) for job in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    return to_job_response(await service.get_job(job_id), service)


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: int, service: JobsServiceDep) -> None:
    await service.delete_job(job_id)


@router.post("/{job_id}/enqueue", response_model=JobResponse, status_code=200)
async def enqueue_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    await service.enqueue_job(job_id)
    return to_job_response(await service.get_job(job_id), service)


@router.post("/{job_id}/resume", response_model=JobResponse, status_code=200)
async def resume_job(job_id: int, service: JobsServiceDep) -> JobResponse:
    await service.resume_job(job_id)
    return to_job_response(await service.get_job(job_id), service)


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: int, service: JobsServiceDep, save_checkpoint: bool = False) -> JobResponse:
    return to_job_response(await service.cancel_job(job_id, save_checkpoint=save_checkpoint), service)


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


@router.get("/{job_id}/samples", response_model=JobSamplesResponse)
async def get_job_samples(job_id: int, service: JobsServiceDep) -> JobSamplesResponse:
    job = await service.get_job(job_id)
    samples = []
    for sample, kind, metadata in service.list_samples(job):
        relative = sample.relative_to(Path(job.output_path)).as_posix() if job.output_path else sample.name
        samples.append(
            JobSampleResponse(
                filename=sample.name,
                path=str(sample),
                url=_sample_url(job_id, relative),
                kind=kind,  # type: ignore[arg-type]
                metadata=metadata,
            )
        )
    return JobSamplesResponse(samples=samples)


@router.get("/{job_id}/sweep-manifest", response_model=SweepManifestResponse)
async def get_sweep_manifest(job_id: int, service: JobsServiceDep) -> SweepManifestResponse:
    job = await service.get_job(job_id)
    manifest = service.get_sweep_manifest(job)
    if manifest is None:
        return SweepManifestResponse(job_id=job_id, total_images=0)
    output_path = Path(job.output_path) if job.output_path else None
    images = [
        ManifestImageEntryResponse(
            index=img.index,
            file=img.file,
            url=_sample_url(job_id, img.file),
            params=img.params,
            grid_position=img.grid_position,
        )
        for img in manifest.images
    ]
    grids = [
        ManifestGridEntryResponse(
            index=grid.index,
            file=grid.file,
            url=_sample_url(job_id, grid.file),
            slice=grid.slice,
            x=ManifestGridAxisResponse(param=grid.x.param, values=grid.x.values),
            y=ManifestGridAxisResponse(param=grid.y.param, values=grid.y.values),
            cells=grid.cells,
            title=grid.title,
        )
        for grid in manifest.grids
    ]
    return SweepManifestResponse(
        version=manifest.version,
        config_id=manifest.config_id,
        job_id=manifest.job_id or job_id,
        total_images=manifest.total_images,
        images=images,
        grids=grids,
    )


@router.get("/{job_id}/sample-file/{file_path:path}")
async def get_job_sample_file(job_id: int, file_path: str, service: JobsServiceDep) -> FileResponse:
    job = await service.get_job(job_id)
    try:
        path = service.sample_file_path(job, file_path)
    except JobOperationNotSupportedError as exc:
        raise exc
    return FileResponse(path)
