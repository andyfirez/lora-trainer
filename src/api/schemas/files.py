"""Pydantic schemas for native file picker."""

from pydantic import BaseModel, Field

from src.services.files.service import PickKind


class PickPathRequest(BaseModel):
    kind: PickKind = PickKind.MODEL
    title: str = Field(default="Select file", min_length=1)
    initial_path: str | None = None


class PickPathResponse(BaseModel):
    path: str
