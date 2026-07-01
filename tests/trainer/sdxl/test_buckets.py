"""Tests for SDXL aspect-ratio bucket assignment."""

import pytest

from src.trainer.sdxl.buckets import (
    apply_center_offset,
    assign_bucket,
    compute_add_time_ids,
    make_bucket_resolutions,
)


def test_make_bucket_resolutions_winx_steps() -> None:
    resos = make_bucket_resolutions((1024, 1024), min_size=512, max_size=1024, divisible=256)
    assert (1024, 1024) in resos
    assert (512, 1024) in resos
    for width, height in resos:
        assert width % 256 == 0
        assert height % 256 == 0


def test_assign_bucket_landscape_not_square() -> None:
    assignment = assign_bucket(
        1216,
        918,
        resolution=1024,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_reso_steps=256,
        bucket_no_upscale=True,
    )
    assert assignment.bucket_width != assignment.bucket_height or (
        assignment.bucket_width != 1024
    )
    assert assignment.bucket_width * assignment.bucket_height <= 1024 * 1024
    assert assignment.bucket_width % 256 == 0
    assert assignment.bucket_height % 256 == 0


def test_assign_bucket_square_source() -> None:
    assignment = assign_bucket(
        1024,
        1024,
        resolution=1024,
        bucket_reso_steps=256,
    )
    assert (assignment.bucket_width, assignment.bucket_height) == (1024, 1024)


def test_assign_bucket_portrait() -> None:
    assignment = assign_bucket(
        918,
        1216,
        resolution=1024,
        bucket_reso_steps=256,
        bucket_no_upscale=True,
    )
    assert assignment.bucket_height >= assignment.bucket_width


def test_bucket_no_upscale_skips_larger_buckets() -> None:
    assignment = assign_bucket(
        256,
        256,
        resolution=1024,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_reso_steps=256,
        bucket_no_upscale=True,
    )
    assert assignment.bucket_width <= 512
    assert assignment.bucket_height <= 512


def test_compute_add_time_ids_kohya_order() -> None:
    assignment = assign_bucket(1216, 918, resolution=1024, bucket_reso_steps=256)
    ids = compute_add_time_ids(assignment)
    assert ids == (
        918,
        1216,
        assignment.crop_y,
        assignment.crop_x,
        assignment.bucket_height,
        assignment.bucket_width,
    )


def test_apply_center_offset_clamps() -> None:
    crop_x, crop_y = apply_center_offset(
        crop_x=0,
        crop_y=0,
        scale_w=2000,
        scale_h=1500,
        bucket_w=1024,
        bucket_h=768,
        center_x=0.0,
        center_y=0.0,
    )
    assert crop_x == 0
    assert crop_y == 0

    crop_x2, crop_y2 = apply_center_offset(
        crop_x=0,
        crop_y=0,
        scale_w=2000,
        scale_h=1500,
        bucket_w=1024,
        bucket_h=768,
        center_x=1.0,
        center_y=1.0,
    )
    assert crop_x2 == 2000 - 1024
    assert crop_y2 == 1500 - 768


def test_assign_bucket_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        assign_bucket(0, 100, resolution=1024)
