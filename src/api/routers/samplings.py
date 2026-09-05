"""Sampling router — create/enqueue/cancel and result access."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.api.dependencies import SamplingServiceDep
from src.api.routers.runnable import RunnableRouterHooks, build_runnable_router
from src.api.schemas.samplings import (
    CreateSamplingRequest,
    PromptTokenCountRequest,
    PromptTokenCountResponse,
    SamplerOption,
    SamplingOptionsResponse,
    SamplingResponse,
    SchedulerOption,
)
from src.db.tables.sampling import Sampling, SamplingKind
from src.sampler.experiment.manifest import ExperimentManifest
from src.sampler.registry.profiles import (
    SamplingProfile,
    sampler_label,
    samplers_for_profile,
    scheduler_label,
    schedulers_for_profile,
)
from src.services.sampling.service import SamplingService
from src.trainer.sdxl.prompt_chunking import CLIP_CONTENT_LENGTH, count_prompt_chunks
from src.trainer.sdxl.tokenizer_loader import load_sdxl_tokenizers

options_router = APIRouter(prefix="/samplings", tags=["samplings"])


@options_router.get("/options", response_model=SamplingOptionsResponse, operation_id="get_samplings_options")
async def get_sampling_options(
    profile: SamplingProfile = Query(default=SamplingProfile.COMFY),
) -> SamplingOptionsResponse:
    samplers = [
        SamplerOption(
            id=spec.id,
            label=sampler_label(spec, profile),
            default_scheduler=spec.default_scheduler,
        )
        for spec in samplers_for_profile(profile)
    ]
    schedulers = [
        SchedulerOption(id=spec.id, label=scheduler_label(spec, profile))
        for spec in schedulers_for_profile(profile)
    ]
    return SamplingOptionsResponse(profile=profile, samplers=samplers, schedulers=schedulers)


@options_router.post("/token-count", response_model=PromptTokenCountResponse, operation_id="count_sampling_prompt_tokens")
async def count_sampling_prompt_tokens(body: PromptTokenCountRequest) -> PromptTokenCountResponse:
    try:
        tokenizer_1, tokenizer_2 = load_sdxl_tokenizers(body.base_model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    counts = count_prompt_chunks(body.prompt, tokenizer_1, tokenizer_2)
    return PromptTokenCountResponse(tokens=counts.tokens, chunks=counts.chunks, window=CLIP_CONTENT_LENGTH)


def _to_response(sampling: Sampling, service: SamplingService) -> SamplingResponse:
    data = sampling.model_dump()
    data["lora_paths"] = service.get_lora_paths(sampling)
    return SamplingResponse.model_validate(data)


def _output_dir(sampling: Sampling) -> Path | None:
    return Path(sampling.output_path) if sampling.output_path else None


async def _create_sampling(service: SamplingService, body: CreateSamplingRequest) -> Sampling:
    sampling = await service.create_sampling(
        name=body.name,
        config=body.config,
        lora_paths=body.lora_paths,
        kind=body.kind,
    )
    if body.enqueue and sampling.id is not None:
        return await service.enqueue_sampling(sampling.id)
    return sampling


router = build_runnable_router(
    prefix="/samplings",
    tags=["samplings"],
    service_dep=SamplingServiceDep,
    response_cls=SamplingResponse,
    create_cls=CreateSamplingRequest,
    hooks=RunnableRouterHooks(
        list_entities=lambda service: service.list_samplings(),
        list_entities_for_kind=lambda service, kind: service.list_samplings(kind=SamplingKind(kind)),
        get_entity=lambda service, entity_id: service.get_sampling(entity_id),
        create_entity=_create_sampling,
        enqueue_entity=lambda service, entity_id: service.enqueue_sampling(entity_id),
        cancel_entity=lambda service, entity_id: service.cancel_sampling(entity_id),
        get_logs=lambda service, entity_id, tail: service.get_logs(entity_id, tail=tail),
        list_samples=lambda service, entity: service.list_samples(entity),
        sample_file_path=lambda service, entity, relative: service.sample_file_path(entity, relative),
        output_dir=_output_dir,
        to_response=_to_response,
    ),
)


@router.get("/{entity_id}/experiment-manifest", response_model=Optional[ExperimentManifest])
async def get_experiment_manifest(entity_id: int, service: SamplingServiceDep) -> ExperimentManifest | None:
    sampling = await service.get_sampling(entity_id)
    return service.get_experiment_manifest(sampling)


@router.get("/{entity_id}/live-preview", operation_id="get_samplings_live_preview")
async def get_live_preview(entity_id: int, service: SamplingServiceDep) -> FileResponse:
    sampling = await service.get_sampling(entity_id)
    return FileResponse(
        service.live_preview_path(sampling),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
