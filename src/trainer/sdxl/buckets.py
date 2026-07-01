"""Aspect-ratio bucket assignment for SDXL training (Kohya-compatible)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketAssignment:
    bucket_width: int
    bucket_height: int
    scale_to_width: int
    scale_to_height: int
    crop_x: int
    crop_y: int
    source_width: int
    source_height: int

    @property
    def bucket_key(self) -> str:
        return f"{self.bucket_width}x{self.bucket_height}"

    def add_time_ids(self) -> tuple[int, int, int, int, int, int]:
        return compute_add_time_ids(self)


def compute_add_time_ids(assignment: BucketAssignment) -> tuple[int, int, int, int, int, int]:
    """SDXL micro-conditioning: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)."""
    return (
        assignment.source_height,
        assignment.source_width,
        assignment.crop_y,
        assignment.crop_x,
        assignment.bucket_height,
        assignment.bucket_width,
    )


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
) -> list[tuple[int, int]]:
    max_width, max_height = max_reso
    max_area = (max_width // divisible) * (max_height // divisible)
    resos: set[tuple[int, int]] = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    step = min_size
    while step <= max_size:
        width = step
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        resos.add((width, height))
        resos.add((height, width))
        step += divisible

    return sorted(resos)


def _native_bucket_size(dimension: int, *, min_size: int, step: int) -> int:
    snapped = (dimension // step) * step
    if snapped < min_size:
        return min_size
    return max(snapped, step)


def _select_bucket_resolution(
    source_width: int,
    source_height: int,
    resolutions: list[tuple[int, int]],
    *,
    min_size: int,
    bucket_reso_steps: int,
    bucket_no_upscale: bool,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not resolutions:
        raise ValueError("No bucket resolutions available")

    aspect = source_width / source_height
    candidates: list[tuple[int, int]] = []
    for bucket_width, bucket_height in resolutions:
        if bucket_no_upscale:
            scale = max(bucket_width / source_width, bucket_height / source_height)
            if scale > 1.0 + 1e-6:
                continue
        candidates.append((bucket_width, bucket_height))

    if not candidates and bucket_no_upscale:
        return (
            _native_bucket_size(source_width, min_size=min_size, step=bucket_reso_steps),
            _native_bucket_size(source_height, min_size=min_size, step=bucket_reso_steps),
        )

    if not candidates:
        candidates = list(resolutions)

    def _score(bucket: tuple[int, int]) -> tuple[float, float]:
        bucket_width, bucket_height = bucket
        bucket_aspect = bucket_width / bucket_height
        aspect_delta = abs(aspect - bucket_aspect)
        pixel_delta = abs(bucket_width * bucket_height - source_width * source_height)
        return (aspect_delta, pixel_delta)

    return min(candidates, key=_score)


def _scale_and_crop(
    source_width: int,
    source_height: int,
    bucket_width: int,
    bucket_height: int,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int]:
    width_scale = bucket_width / source_width
    height_scale = bucket_height / source_height
    max_scale = max(width_scale, height_scale)
    scale_to_width = max(int(math.ceil(source_width * max_scale)), bucket_width)
    scale_to_height = max(int(math.ceil(source_height * max_scale)), bucket_height)
    crop_x, crop_y = apply_center_offset(
        crop_x=(scale_to_width - bucket_width) // 2,
        crop_y=(scale_to_height - bucket_height) // 2,
        scale_w=scale_to_width,
        scale_h=scale_to_height,
        bucket_w=bucket_width,
        bucket_h=bucket_height,
        center_x=center_x,
        center_y=center_y,
    )
    return scale_to_width, scale_to_height, crop_x, crop_y


def apply_center_offset(
    crop_x: int,
    crop_y: int,
    scale_w: int,
    scale_h: int,
    bucket_w: int,
    bucket_h: int,
    center_x: float,
    center_y: float,
) -> tuple[int, int]:
    if scale_w <= bucket_w:
        crop_x = 0
    else:
        half = bucket_w / 2.0
        min_cx = half / scale_w
        max_cx = 1.0 - half / scale_w
        cx = min(max(center_x, min_cx), max_cx)
        crop_x = int(round(cx * scale_w - half))
        crop_x = max(0, min(crop_x, scale_w - bucket_w))

    if scale_h <= bucket_h:
        crop_y = 0
    else:
        half = bucket_h / 2.0
        min_cy = half / scale_h
        max_cy = 1.0 - half / scale_h
        cy = min(max(center_y, min_cy), max_cy)
        crop_y = int(round(cy * scale_h - half))
        crop_y = max(0, min(crop_y, scale_h - bucket_h))

    return crop_x, crop_y


def assign_bucket(
    source_width: int,
    source_height: int,
    *,
    resolution: int,
    min_bucket_reso: int = 512,
    max_bucket_reso: int = 2048,
    bucket_reso_steps: int = 64,
    bucket_no_upscale: bool = True,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> BucketAssignment:
    resolutions = make_bucket_resolutions(
        (resolution, resolution),
        min_size=min_bucket_reso,
        max_size=min(resolution, max_bucket_reso),
        divisible=bucket_reso_steps,
    )
    bucket_width, bucket_height = _select_bucket_resolution(
        source_width,
        source_height,
        resolutions,
        min_size=min_bucket_reso,
        bucket_reso_steps=bucket_reso_steps,
        bucket_no_upscale=bucket_no_upscale,
    )
    scale_to_width, scale_to_height, crop_x, crop_y = _scale_and_crop(
        source_width,
        source_height,
        bucket_width,
        bucket_height,
        center_x,
        center_y,
    )
    return BucketAssignment(
        bucket_width=bucket_width,
        bucket_height=bucket_height,
        scale_to_width=scale_to_width,
        scale_to_height=scale_to_height,
        crop_x=crop_x,
        crop_y=crop_y,
        source_width=source_width,
        source_height=source_height,
    )


def assignment_from_stored(
    *,
    source_width: int,
    source_height: int,
    bucket_width: int,
    bucket_height: int,
    scale_to_width: int,
    scale_to_height: int,
    crop_x: int,
    crop_y: int,
) -> BucketAssignment:
    return BucketAssignment(
        bucket_width=bucket_width,
        bucket_height=bucket_height,
        scale_to_width=scale_to_width,
        scale_to_height=scale_to_height,
        crop_x=crop_x,
        crop_y=crop_y,
        source_width=source_width,
        source_height=source_height,
    )
