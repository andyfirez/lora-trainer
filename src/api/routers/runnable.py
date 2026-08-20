"""Shared FastAPI helpers for LoRA and Sampling runnable routes."""

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api.schemas.job_logs import JobLogsResponse
from src.api.schemas.runnable import RunnableSampleResponse, RunnableSamplesResponse

TEntity = TypeVar("TEntity")
TService = TypeVar("TService")
TResponse = TypeVar("TResponse")
TCreate = TypeVar("TCreate")


def sample_file_url(prefix: str, entity_id: int, relative: str) -> str:
    resource = prefix.strip("/")
    return f"/{resource}/{entity_id}/sample-file/{relative.replace(chr(92), '/')}"


def build_sample_responses(
    entity_id: int,
    prefix: str,
    output_dir: Path | None,
    samples: Iterable[tuple[Path, str, dict]],
) -> RunnableSamplesResponse:
    if output_dir is None:
        return RunnableSamplesResponse()
    resolved = output_dir.resolve()
    items = [
        RunnableSampleResponse(
            filename=sample.name,
            path=str(sample),
            url=sample_file_url(prefix, entity_id, sample.relative_to(resolved).as_posix()),
            kind=kind,
            metadata=metadata,
        )
        for sample, kind, metadata in samples
    ]
    return RunnableSamplesResponse(samples=items)


@dataclass(frozen=True)
class RunnableRouterHooks(Generic[TEntity, TService, TResponse, TCreate]):
    list_entities: Callable[[TService], Awaitable[Sequence[TEntity]]]
    get_entity: Callable[[TService, int], Awaitable[TEntity]]
    create_entity: Callable[[TService, TCreate], Awaitable[TEntity]]
    enqueue_entity: Callable[[TService, int], Awaitable[TEntity]]
    get_logs: Callable[[TService, int, int], Awaitable[list[str]]]
    list_samples: Callable[[TService, TEntity], Sequence[tuple[Path, str, dict]]]
    sample_file_path: Callable[[TService, TEntity, str], Path]
    output_dir: Callable[[TEntity], Path | None]
    to_response: Callable[[TEntity, TService], TResponse]
    cancel_entity: Callable[[TService, int], Awaitable[TEntity]] | None = None


def build_runnable_router(
    *,
    prefix: str,
    tags: list[str],
    service_dep: Any,
    response_cls: type[TResponse],
    create_cls: type[TCreate],
    hooks: RunnableRouterHooks[TEntity, TService, TResponse, TCreate],
) -> APIRouter:
    """Register shared list/create/get/enqueue/logs/samples routes; extra routes attach to the result."""
    router = APIRouter(prefix=prefix, tags=tags)
    slug = prefix.strip("/")

    @router.get("/", response_model=list[response_cls], operation_id=f"list_{slug}")
    async def list_items(service: service_dep) -> list[TResponse]:
        entities = await hooks.list_entities(service)
        return [hooks.to_response(entity, service) for entity in entities]

    @router.post("/", response_model=response_cls, status_code=201, operation_id=f"create_{slug}")
    async def create_item(body: create_cls, service: service_dep) -> TResponse:
        entity = await hooks.create_entity(service, body)
        return hooks.to_response(entity, service)

    @router.get("/{entity_id}", response_model=response_cls, operation_id=f"get_{slug}")
    async def get_item(entity_id: int, service: service_dep) -> TResponse:
        entity = await hooks.get_entity(service, entity_id)
        return hooks.to_response(entity, service)

    @router.post("/{entity_id}/enqueue", response_model=response_cls, operation_id=f"enqueue_{slug}")
    async def enqueue_item(entity_id: int, service: service_dep) -> TResponse:
        entity = await hooks.enqueue_entity(service, entity_id)
        return hooks.to_response(entity, service)

    if hooks.cancel_entity is not None:
        cancel_fn = hooks.cancel_entity

        @router.post("/{entity_id}/cancel", response_model=response_cls, operation_id=f"cancel_{slug}")
        async def cancel_item(entity_id: int, service: service_dep) -> TResponse:
            entity = await cancel_fn(service, entity_id)
            return hooks.to_response(entity, service)

    @router.get("/{entity_id}/logs", response_model=JobLogsResponse, operation_id=f"get_{slug}_logs")
    async def get_logs(entity_id: int, service: service_dep, tail: int = 500) -> JobLogsResponse:
        lines = await hooks.get_logs(service, entity_id, tail)
        return JobLogsResponse(lines=lines)

    @router.get(
        "/{entity_id}/samples",
        response_model=RunnableSamplesResponse,
        operation_id=f"get_{slug}_samples",
    )
    async def get_samples(entity_id: int, service: service_dep) -> RunnableSamplesResponse:
        entity = await hooks.get_entity(service, entity_id)
        return build_sample_responses(
            entity_id,
            slug,
            hooks.output_dir(entity),
            hooks.list_samples(service, entity),
        )

    @router.get("/{entity_id}/sample-file/{file_path:path}", operation_id=f"get_{slug}_sample_file")
    async def get_sample_file(entity_id: int, file_path: str, service: service_dep) -> FileResponse:
        entity = await hooks.get_entity(service, entity_id)
        return FileResponse(hooks.sample_file_path(service, entity, file_path))

    return router
