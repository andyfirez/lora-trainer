"""Pydantic schemas for the /loras API."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.api.schemas.runnable import RunnableResponse
from src.db.tables.runnable_mixin import RunnableStatus

_RESUMABLE_STATUSES = (RunnableStatus.FAILED, RunnableStatus.CANCELLED, RunnableStatus.ORPHAN)


class LoraResponse(RunnableResponse):
    config_yaml: Optional[str] = None
    relative_path: str
    weights_relpath: str
    base_model_name: str
    resolved_work_dir: str = ""
    resolved_weights_path: str = ""
    path_missing: bool = True
    can_resume: bool = False

    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    progress_loss: Optional[float] = None
    progress_avr_loss: Optional[float] = None
    progress_epoch: Optional[int] = None
    progress_epoch_total: Optional[int] = None
    cache_progress_step: Optional[int] = None
    cache_progress_total: Optional[int] = None

    last_checkpoint_path: Optional[str] = None
    last_checkpoint_epoch: Optional[int] = None
    last_checkpoint_step: Optional[int] = None
    resume_checkpoint_path: Optional[str] = None
    resume_from_epoch: Optional[int] = None
    resume_from_step: Optional[int] = None
    save_checkpoint_requested: bool = False

    @model_validator(mode="before")
    @classmethod
    def _populate_resolved_paths(cls, data: object) -> object:
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
        data.setdefault(
            "can_resume",
            data.get("status") in _RESUMABLE_STATUSES and bool(data.get("last_checkpoint_path")),
        )
        return data


class LoraSampleResponse(BaseModel):
    filename: str
    path: str
    url: str
    kind: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class LoraSamplesResponse(BaseModel):
    samples: list[LoraSampleResponse] = Field(default_factory=list)


class CreateLoraRequest(BaseModel):
    name: str
    config_yaml: str


class ReproduceLoraRequest(BaseModel):
    name: Optional[str] = None
    enqueue: bool = False
