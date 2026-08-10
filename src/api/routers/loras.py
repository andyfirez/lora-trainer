"""LoRA training router — create/enqueue/resume/cancel and artifact access."""

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.dependencies import LoraServiceDep
from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.job_loss import JobLossResponse
from src.api.schemas.loras import (
    CreateLoraRequest,
    LoraResponse,
    LoraSampleResponse,
    LoraSamplesResponse,
    ReproduceLoraRequest,
)
from src.services.loras.exceptions import LoraNotFoundError
from src.services.loras.paths import resolve_weights_path, resolve_work_dir
from src.services.runnable.exceptions import RunnableOperationNotSupportedError

router = APIRouter(prefix="/loras", tags=["loras"])


class CancelLoraRequest(BaseModel):
    save_checkpoint: bool = False


def _sample_url(lora_id: int, relative: str) -> str:
    return f"/loras/{lora_id}/sample-file/{relative.replace(chr(92), '/')}"


@router.get("/", response_model=list[LoraResponse])
async def list_loras(service: LoraServiceDep) -> list[LoraResponse]:
    loras = await service.list_loras()
    return [LoraResponse.model_validate(lora) for lora in loras]


@router.post("/", response_model=LoraResponse, status_code=201)
async def create_lora(body: CreateLoraRequest, service: LoraServiceDep) -> LoraResponse:
    lora = await service.create_lora(name=body.name, config_yaml=body.config_yaml)
    return LoraResponse.model_validate(lora)


@router.get("/{lora_id}", response_model=LoraResponse)
async def get_lora(lora_id: int, service: LoraServiceDep) -> LoraResponse:
    lora = await service.get_lora(lora_id)
    return LoraResponse.model_validate(lora)


@router.post("/{lora_id}/enqueue", response_model=LoraResponse)
async def enqueue_lora(lora_id: int, service: LoraServiceDep) -> LoraResponse:
    lora = await service.enqueue_lora(lora_id)
    return LoraResponse.model_validate(lora)


@router.post("/{lora_id}/resume", response_model=LoraResponse)
async def resume_lora(lora_id: int, service: LoraServiceDep) -> LoraResponse:
    lora = await service.resume_lora(lora_id)
    return LoraResponse.model_validate(lora)


@router.post("/{lora_id}/cancel", response_model=LoraResponse)
async def cancel_lora(lora_id: int, body: CancelLoraRequest, service: LoraServiceDep) -> LoraResponse:
    lora = await service.cancel_lora(lora_id, save_checkpoint=body.save_checkpoint)
    return LoraResponse.model_validate(lora)


@router.get("/{lora_id}/logs", response_model=JobLogsResponse)
async def get_lora_logs(lora_id: int, service: LoraServiceDep, tail: int = 500) -> JobLogsResponse:
    lines = await service.get_logs(lora_id, tail=tail)
    return JobLogsResponse(lines=lines)


@router.get("/{lora_id}/loss", response_model=JobLossResponse)
async def get_lora_loss(
    lora_id: int,
    service: LoraServiceDep,
    key: str = "loss/loss",
    limit: int = 2000,
    since_step: int | None = None,
    stride: int = 1,
) -> JobLossResponse:
    return await service.get_loss(lora_id, key=key, limit=limit, since_step=since_step, stride=stride)


@router.get("/{lora_id}/samples", response_model=LoraSamplesResponse)
async def get_lora_samples(lora_id: int, service: LoraServiceDep) -> LoraSamplesResponse:
    lora = await service.get_lora(lora_id)
    output_dir = resolve_work_dir(lora)
    samples = []
    for sample, kind, metadata in service.list_samples(lora):
        relative = sample.relative_to(output_dir).as_posix()
        samples.append(
            LoraSampleResponse(
                filename=sample.name,
                path=str(sample),
                url=_sample_url(lora_id, relative),
                kind=kind,  # type: ignore[arg-type]
                metadata=metadata,
            )
        )
    return LoraSamplesResponse(samples=samples)


@router.get("/{lora_id}/weights")
async def download_lora_weights(lora_id: int, service: LoraServiceDep) -> FileResponse:
    lora = await service.get_lora(lora_id)
    path = resolve_weights_path(lora)
    if not path.is_file():
        raise LoraNotFoundError(lora_id)
    return FileResponse(path, filename=path.name)


@router.get("/{lora_id}/sample-file/{file_path:path}")
async def get_lora_sample_file(lora_id: int, file_path: str, service: LoraServiceDep) -> FileResponse:
    lora = await service.get_lora(lora_id)
    base = resolve_work_dir(lora).resolve()
    target = (base / file_path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise RunnableOperationNotSupportedError("Lora", lora_id, "sample file")
    return FileResponse(target)


@router.post("/{lora_id}/reproduce", response_model=LoraResponse, status_code=201)
async def reproduce_lora(lora_id: int, body: ReproduceLoraRequest, service: LoraServiceDep) -> LoraResponse:
    source = await service.get_lora(lora_id)
    name = body.name or f"{source.name}-copy"
    lora = await service.reproduce(lora_id, name=name)
    if body.enqueue and lora.id is not None:
        lora = await service.enqueue_lora(lora.id)
    return LoraResponse.model_validate(lora)
