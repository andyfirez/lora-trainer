"""Managed storage browse endpoints."""

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.storage import StorageBrowseResponse, StorageEntryResponse
from src.services.storage.browse import StorageBrowseService
from src.storage.paths import StorageKind, StoragePaths

router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/browse", response_model=StorageBrowseResponse)
async def browse_storage(
    kind: StorageKind = Query(...),
    relative_path: str = Query(default=""),
) -> StorageBrowseResponse:
    try:
        StoragePaths.validate_relative_path(kind, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    browse = StorageBrowseService()
    if kind == StorageKind.BASE_MODELS:
        entries = browse.list_model_entries(relative_path)
    elif kind == StorageKind.LORA:
        entries = browse.list_lora_entries(relative_path)
    else:
        entries = browse.list_entries(kind, relative_path)

    return StorageBrowseResponse(
        kind=kind,
        root=str(StoragePaths.root_for(kind)),
        relative_path=relative_path.strip().strip("/\\"),
        entries=[
            StorageEntryResponse(
                name=e.name,
                relative_path=e.relative_path,
                is_dir=e.is_dir,
                is_lora_work_dir=e.is_lora_work_dir,
            )
            for e in entries
        ],
    )
