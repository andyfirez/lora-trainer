"""Dataset caption, tag, and autotag routes."""

from fastapi import APIRouter, Query

from src.api.dependencies import DatasetDep, DatasetsServiceDep
from src.api.schemas.datasets import (
    AutotagRequest,
    AutotagStatusResponse,
    BulkTagRequest,
    BulkTagResponse,
    CaptionResponse,
    CaptionUpdateRequest,
    TagStatResponse,
    TagStatsResponse,
)
from src.services.tagging.manager import TaggingTaskState

router = APIRouter()


def _autotag_status_response(state: TaggingTaskState) -> AutotagStatusResponse:
    return AutotagStatusResponse(
        status=state.status.value,
        current=state.current,
        total=state.total,
        message=state.message,
        error=state.error,
    )


@router.get("/{dataset_id}/captions/{filename}", response_model=CaptionResponse)
async def get_caption(
    dataset: DatasetDep,
    filename: str,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> CaptionResponse:
    tags = service.get_tags(dataset, filename, caption_extension)
    return CaptionResponse(filename=filename, tags=tags)


@router.put("/{dataset_id}/captions/{filename}", response_model=CaptionResponse)
async def update_caption(
    dataset: DatasetDep,
    filename: str,
    body: CaptionUpdateRequest,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> CaptionResponse:
    tags = service.update_tags(dataset, filename, body.tags, caption_extension)
    return CaptionResponse(filename=filename, tags=tags)


@router.get("/{dataset_id}/tags/stats", response_model=TagStatsResponse)
async def get_tag_stats(
    dataset: DatasetDep,
    service: DatasetsServiceDep,
    caption_extension: str = Query(default=".txt"),
) -> TagStatsResponse:
    stats = service.get_tag_stats(dataset, caption_extension)
    return TagStatsResponse(tags=[TagStatResponse(tag=stat.tag, count=stat.count) for stat in stats])


@router.post("/{dataset_id}/tags/bulk-add", response_model=BulkTagResponse)
async def bulk_add_tag(
    dataset: DatasetDep,
    body: BulkTagRequest,
    service: DatasetsServiceDep,
) -> BulkTagResponse:
    updated = service.bulk_add_tag(
        dataset,
        body.tag,
        body.filenames,
        body.caption_extension,
    )
    return BulkTagResponse(updated_count=updated)


@router.post("/{dataset_id}/tags/bulk-remove", response_model=BulkTagResponse)
async def bulk_remove_tag(
    dataset: DatasetDep,
    body: BulkTagRequest,
    service: DatasetsServiceDep,
) -> BulkTagResponse:
    updated = service.bulk_remove_tag(
        dataset,
        body.tag,
        body.filenames,
        body.caption_extension,
    )
    return BulkTagResponse(updated_count=updated)


@router.post("/{dataset_id}/autotag", response_model=AutotagStatusResponse, status_code=202)
async def autotag_dataset(
    body: AutotagRequest,
    dataset: DatasetDep,
    service: DatasetsServiceDep,
) -> AutotagStatusResponse:
    state = service.start_autotag(dataset, **body.model_dump())
    return _autotag_status_response(state)


@router.get("/{dataset_id}/autotag/status", response_model=AutotagStatusResponse)
async def get_autotag_status(dataset: DatasetDep, service: DatasetsServiceDep) -> AutotagStatusResponse:
    return _autotag_status_response(service.get_autotag_status(dataset))
