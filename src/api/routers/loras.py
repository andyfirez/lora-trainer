"""LoRA training router — create/enqueue/resume/cancel and artifact access."""

from src.api.dependencies import LoraServiceDep
from src.api.routers.runnable import RunnableRouterHooks, build_runnable_router
from src.api.schemas.job_loss import JobLossResponse
from src.api.schemas.loras import (
    CancelLoraRequest,
    CreateLoraRequest,
    LoraResponse,
    ReproduceLoraRequest,
)
from src.services.loras.paths import resolve_sample_base_dir

router = build_runnable_router(
    prefix="/loras",
    tags=["loras"],
    service_dep=LoraServiceDep,
    response_cls=LoraResponse,
    create_cls=CreateLoraRequest,
    hooks=RunnableRouterHooks(
        list_entities=lambda service: service.list_loras(),
        get_entity=lambda service, entity_id: service.get_lora(entity_id),
        create_entity=lambda service, body: service.create_lora(
            name=body.name, config_yaml=body.config_yaml
        ),
        enqueue_entity=lambda service, entity_id: service.enqueue_lora(entity_id),
        get_logs=lambda service, entity_id, tail: service.get_logs(entity_id, tail=tail),
        list_samples=lambda service, entity: service.list_samples(entity),
        sample_file_path=lambda service, entity, relative: service.sample_file_path(entity, relative),
        output_dir=resolve_sample_base_dir,
        to_response=lambda entity, _service: LoraResponse.model_validate(entity),
    ),
)


@router.post("/{entity_id}/resume", response_model=LoraResponse)
async def resume_lora(entity_id: int, service: LoraServiceDep) -> LoraResponse:
    lora = await service.resume_lora(entity_id)
    return LoraResponse.model_validate(lora)


@router.post("/{entity_id}/cancel", response_model=LoraResponse)
async def cancel_lora(entity_id: int, body: CancelLoraRequest, service: LoraServiceDep) -> LoraResponse:
    lora = await service.cancel_lora(entity_id, save_checkpoint=body.save_checkpoint)
    return LoraResponse.model_validate(lora)


@router.get("/{entity_id}/loss", response_model=JobLossResponse)
async def get_lora_loss(
    entity_id: int,
    service: LoraServiceDep,
    key: str = "loss/loss",
    limit: int = 2000,
    since_step: int | None = None,
    stride: int = 1,
) -> JobLossResponse:
    return await service.get_loss(entity_id, key=key, limit=limit, since_step=since_step, stride=stride)


@router.post("/{entity_id}/reproduce", response_model=LoraResponse, status_code=201)
async def reproduce_lora(entity_id: int, body: ReproduceLoraRequest, service: LoraServiceDep) -> LoraResponse:
    source = await service.get_lora(entity_id)
    name = body.name or f"{source.name}-copy"
    lora = await service.reproduce(entity_id, name=name)
    if body.enqueue and lora.id is not None:
        lora = await service.enqueue_lora(lora.id)
    return LoraResponse.model_validate(lora)
