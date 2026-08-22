"""Pydantic schemas for the /samplings API."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.api.schemas.runnable import RunnableResponse


class SamplingResponse(RunnableResponse):
    config: dict[str, Any]
    lora_paths: list[str] = Field(default_factory=list)
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    progress_status: Optional[str] = None


class CreateSamplingRequest(BaseModel):
    name: str
    config: dict[str, Any]
    lora_paths: Optional[list[str]] = None
