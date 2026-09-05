"""Pydantic schemas for the /samplings API."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.api.schemas.runnable import RunnableResponse
from src.db.tables.sampling import SamplingKind
from src.sampler.registry.profiles import SamplingProfile


class SamplingResponse(RunnableResponse):
    kind: SamplingKind
    config: dict[str, Any]
    lora_paths: list[str] = Field(default_factory=list)
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    progress_status: Optional[str] = None


class CreateSamplingRequest(BaseModel):
    name: str
    config: dict[str, Any]
    lora_paths: Optional[list[str]] = None
    kind: SamplingKind = SamplingKind.PLAYGROUND


class SamplerOption(BaseModel):
    id: str
    label: str
    default_scheduler: str


class SchedulerOption(BaseModel):
    id: str
    label: str


class SamplingOptionsResponse(BaseModel):
    profile: SamplingProfile
    samplers: list[SamplerOption]
    schedulers: list[SchedulerOption]


class PromptTokenCountRequest(BaseModel):
    prompt: str
    base_model_name: str


class PromptTokenCountResponse(BaseModel):
    tokens: int
    chunks: int
    window: int = 75
