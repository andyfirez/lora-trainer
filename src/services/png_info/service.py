"""Inspect uploaded images for embedded generation metadata."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from src.services.png_info.exceptions import InvalidImageError
from src.services.png_info.names import split_combined_sampler_name
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
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            geninfo, items = read_info_from_image(image)
            info = geninfo or ""
            merged_items: dict[str, str] = {}
            if info:
                merged_items["parameters"] = info
            merged_items.update(items)

            parameters = parse_generation_parameters(info) if info else {}
            if parameters:
                sampler_field = parameters.get("Sampler")
                schedule_field = parameters.get("Schedule type")
                if isinstance(sampler_field, str):
                    sampler_id, scheduler_id = split_combined_sampler_name(sampler_field)
                    if sampler_id:
                        parameters["sampler_id"] = sampler_id
                    if scheduler_id:
                        parameters["scheduler_id"] = scheduler_id
                    elif isinstance(schedule_field, str):
                        _, from_schedule = split_combined_sampler_name(schedule_field)
                        if from_schedule:
                            parameters["scheduler_id"] = from_schedule
                        else:
                            from src.services.png_info.names import lookup_scheduler_id

                            resolved = lookup_scheduler_id(schedule_field)
                            if resolved:
                                parameters["scheduler_id"] = resolved

            return PngInfoResult(
                info=info,
                items=merged_items,
                parameters=parameters,
                width=image.width,
                height=image.height,
                preview_base64=_build_preview_base64(image),
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError() from exc
