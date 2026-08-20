"""Sampling router — create/enqueue/cancel and result access."""

from pathlib import Path
from typing import Optional

from src.api.dependencies import SamplingServiceDep
from src.api.routers.runnable import RunnableRouterHooks, build_runnable_router
from src.api.schemas.samplings import CreateSamplingRequest, SamplingResponse
from src.db.tables.sampling import Sampling
from src.sampler.sweep.manifest import SweepManifest
from src.services.sampling.service import SamplingService


def _to_response(sampling: Sampling, service: SamplingService) -> SamplingResponse:
    data = sampling.model_dump()
    data["lora_paths"] = service.get_lora_paths(sampling)
    return SamplingResponse.model_validate(data)


def _output_dir(sampling: Sampling) -> Path | None:
    return Path(sampling.output_path) if sampling.output_path else None


router = build_runnable_router(
    prefix="/samplings",
    tags=["samplings"],
    service_dep=SamplingServiceDep,
    response_cls=SamplingResponse,
    create_cls=CreateSamplingRequest,
    hooks=RunnableRouterHooks(
        list_entities=lambda service: service.list_samplings(),
        get_entity=lambda service, entity_id: service.get_sampling(entity_id),
        create_entity=lambda service, body: service.create_sampling(
            name=body.name, config_yaml=body.config_yaml, lora_paths=body.lora_paths
        ),
        enqueue_entity=lambda service, entity_id: service.enqueue_sampling(entity_id),
        cancel_entity=lambda service, entity_id: service.cancel_sampling(entity_id),
        get_logs=lambda service, entity_id, tail: service.get_logs(entity_id, tail=tail),
        list_samples=lambda service, entity: service.list_samples(entity),
        sample_file_path=lambda service, entity, relative: service.sample_file_path(entity, relative),
        output_dir=_output_dir,
        to_response=_to_response,
    ),
)


@router.get("/{entity_id}/sweep-manifest", response_model=Optional[SweepManifest])
async def get_sweep_manifest(entity_id: int, service: SamplingServiceDep) -> SweepManifest | None:
    sampling = await service.get_sampling(entity_id)
    return service.get_sweep_manifest(sampling)
