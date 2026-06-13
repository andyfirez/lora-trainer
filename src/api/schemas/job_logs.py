"""Pydantic schemas for job log endpoints."""

from pydantic import BaseModel, Field


class JobLogsResponse(BaseModel):
    lines: list[str] = Field(default_factory=list)
