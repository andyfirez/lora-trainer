"""Inspect uploaded images for embedded generation metadata."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image

from src.services.png_info.parser import parse_generation_parameters
from src.services.png_info.reader import read_info_from_image

PREVIEW_MAX_SIZE = 512


@dataclass(frozen=True)
class PngInfoResult:
    info: str
    items: dict[str, str]
    parameters: dict[str, str | int]
    width: int
    height: int
    preview_base64: str | None


def _build_preview_base64(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.Resampling.LANCZOS)
    if preview.mode not in ("RGB", "L"):
        preview = preview.convert("RGB")
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def inspect_image_bytes(data: bytes) -> PngInfoResult:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        geninfo, items = read_info_from_image(image)
        info = geninfo or ""
        merged_items: dict[str, str] = {}
        if info:
            merged_items["parameters"] = info
        merged_items.update(items)

        parameters = parse_generation_parameters(info) if info else {}

        return PngInfoResult(
            info=info,
            items=merged_items,
            parameters=parameters,
            width=image.width,
            height=image.height,
            preview_base64=_build_preview_base64(image),
        )
