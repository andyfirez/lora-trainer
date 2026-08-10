"""Pydantic schemas for storage browse API."""

from pydantic import BaseModel

from src.storage.paths import StorageKind


class StorageEntryResponse(BaseModel):
    name: str
    relative_path: str
    is_dir: bool
    is_lora_work_dir: bool = False


class StorageBrowseResponse(BaseModel):
    kind: StorageKind
    root: str
    relative_path: str
    entries: list[StorageEntryResponse]
