"""Pydantic schemas for the /samplings API."""

from typing import Optional

from pydantic import BaseModel, Field

from src.api.schemas.runnable import RunnableResponse


class SamplingResponse(RunnableResponse):
    config_yaml: str
    lora_paths: list[str] = Field(default_factory=list)
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    progress_status: Optional[str] = None


class CreateSamplingRequest(BaseModel):
    name: str
    config_yaml: str
    lora_paths: Optional[list[str]] = None


class SamplingSampleResponse(BaseModel):
    filename: str
    path: str
    url: str
    kind: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SamplingSamplesResponse(BaseModel):
    samples: list[SamplingSampleResponse] = Field(default_factory=list)
