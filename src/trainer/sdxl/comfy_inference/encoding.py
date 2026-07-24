"""SDXL conditioning via vendored Comfy CLIP."""

from __future__ import annotations

from typing import Any


def build_sdxl_add_dict(
    *,
    width: int,
    height: int,
    reference_add_time_ids: tuple[float, ...] | None = None,
) -> dict[str, float]:
    if reference_add_time_ids is not None:
        if len(reference_add_time_ids) != 6:
            raise ValueError(
                f"reference_add_time_ids must have 6 elements, got {len(reference_add_time_ids)}"
            )
        source_h, source_w, crop_h, crop_w, target_h, target_w = reference_add_time_ids
        return {
            "height": float(source_h),
            "width": float(source_w),
            "crop_h": float(crop_h),
            "crop_w": float(crop_w),
            "target_height": float(target_h),
            "target_width": float(target_w),
        }
    return {
        "height": float(height),
        "width": float(width),
        "crop_h": 0.0,
        "crop_w": 0.0,
        "target_height": float(height),
        "target_width": float(width),
    }


def encode_sdxl_conditioning(
    clip: Any,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    reference_add_time_ids: tuple[float, ...] | None = None,
) -> tuple[list, list]:
    add_dict = build_sdxl_add_dict(
        width=width,
        height=height,
        reference_add_time_ids=reference_add_time_ids,
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt), add_dict=add_dict)
    negative = clip.encode_from_tokens_scheduled(
        clip.tokenize(negative_prompt or ""),
        add_dict=add_dict,
    )
    return positive, negative
