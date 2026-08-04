"""PNG Info API schemas."""

from pydantic import BaseModel, Field


class PngInfoResponse(BaseModel):
    info: str = Field(description="Raw generation info / parameters string")
    items: dict[str, str] = Field(description="All metadata chunks, parameters first when present")
    parameters: dict[str, str | int] = Field(description="Parsed generation parameters")
    width: int
    height: int
    preview_base64: str | None = Field(default=None, description="JPEG preview as data URL")
