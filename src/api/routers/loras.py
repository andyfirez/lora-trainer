"""Trained LoRA catalog router."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api.converters import to_job_response
from src.api.dependencies import JobsServiceDep, TrainedLoraServiceDep
from src.api.schemas.jobs import JobResponse
from src.api.schemas.loras import (
    ReproduceTrainedLoraRequest,
    TrainedLoraResponse,
    TrainedLoraSampleResponse,
    TrainedLoraSamplesResponse,
)
from src.services.jobs.exceptions import JobOperationNotSupportedError
from src.services.jobs.samples import list_samples_for_output_dir
from src.services.loras.exceptions import TrainedLoraNotFoundError
from src.services.loras.paths import resolve_weights_path, resolve_work_dir

router = APIRouter(prefix="/loras", tags=["loras"])


def _sample_url(lora_id: int, relative: str) -> str:
    return f"/loras/{lora_id}/sample-file/{relative.replace(chr(92), '/')}"


@router.get("/", response_model=list[TrainedLoraResponse])
async def list_loras(service: TrainedLoraServiceDep) -> list[TrainedLoraResponse]:
    loras = await service.list_loras()
    return [TrainedLoraResponse.model_validate(lora) for lora in loras]


@router.get("/{lora_id}", response_model=TrainedLoraResponse)
async def get_lora(lora_id: int, service: TrainedLoraServiceDep) -> TrainedLoraResponse:
    lora = await service.get_lora(lora_id)
    return TrainedLoraResponse.model_validate(lora)


@router.get("/{lora_id}/samples", response_model=TrainedLoraSamplesResponse)
async def get_lora_samples(lora_id: int, service: TrainedLoraServiceDep) -> TrainedLoraSamplesResponse:
    lora = await service.get_lora(lora_id)
    output_dir = resolve_work_dir(lora)
    samples = []
    for sample, kind, metadata in list_samples_for_output_dir(output_dir):
        relative = sample.relative_to(output_dir).as_posix()
        samples.append(
            TrainedLoraSampleResponse(
                filename=sample.name,
                path=str(sample),
                url=_sample_url(lora_id, relative),
                kind=kind,  # type: ignore[arg-type]
                metadata=metadata,
            )
        )
    return TrainedLoraSamplesResponse(samples=samples)


@router.get("/{lora_id}/weights")
async def download_lora_weights(lora_id: int, service: TrainedLoraServiceDep) -> FileResponse:
    lora = await service.get_lora(lora_id)
    path = resolve_weights_path(lora)
    if not path.is_file():
        raise TrainedLoraNotFoundError(lora_id)
    return FileResponse(path, filename=path.name)


@router.get("/{lora_id}/sample-file/{file_path:path}")
async def get_lora_sample_file(
    lora_id: int,
    file_path: str,
    service: TrainedLoraServiceDep,
) -> FileResponse:
    lora = await service.get_lora(lora_id)
    base = resolve_work_dir(lora).resolve()
    target = (base / file_path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise JobOperationNotSupportedError(lora.job_id or lora_id, "sample file")
    return FileResponse(target)


@router.post("/{lora_id}/reproduce", response_model=JobResponse, status_code=201)
async def reproduce_lora(
    lora_id: int,
    body: ReproduceTrainedLoraRequest,
    lora_service: TrainedLoraServiceDep,
    jobs_service: JobsServiceDep,
) -> JobResponse:
    job = await lora_service.reproduce(lora_id, name=body.name)
    if body.enqueue and job.id is not None:
        await jobs_service.enqueue_job(job.id)
        job = await jobs_service.get_job(job.id)
    return to_job_response(job, jobs_service)
