"""Pydantic schemas for trained LoRA API endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TrainedLoraResponse(BaseModel):
    id: int
    name: str
    relative_path: str
    weights_relpath: str
    resolved_work_dir: str
    resolved_weights_path: str
    path_missing: bool = False
    job_id: Optional[int] = None
    config_id: Optional[int] = None
    config_yaml: Optional[str] = None
    base_model_name: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def populate_paths(cls, data: object) -> object:
        if not isinstance(data, dict):
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            else:
                return data
        from src.storage.paths import StorageKind, StoragePaths

        relative_path = data.get("relative_path", "")
        weights_relpath = data.get("weights_relpath", "")
        missing = True
        work_dir = ""
        weights = ""
        try:
            work_dir_path = StoragePaths.resolve(StorageKind.LORA, relative_path)
            weights_path = StoragePaths.resolve(StorageKind.LORA, weights_relpath)
            missing = not work_dir_path.is_dir() or not weights_path.is_file()
            if not missing:
                work_dir = str(work_dir_path)
                weights = str(weights_path)
        except (ValueError, OSError):
            missing = True
        data.setdefault("resolved_work_dir", work_dir)
        data.setdefault("resolved_weights_path", weights)
        data.setdefault("path_missing", missing)
        return data


class TrainedLoraSampleResponse(BaseModel):
    filename: str
    path: str
    url: str
    kind: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class TrainedLoraSamplesResponse(BaseModel):
    samples: list[TrainedLoraSampleResponse] = Field(default_factory=list)


class ReproduceTrainedLoraRequest(BaseModel):
    name: Optional[str] = None
    enqueue: bool = False
