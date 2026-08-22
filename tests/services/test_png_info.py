"""Tests for PNG metadata inspection."""

from __future__ import annotations

import io

import pytest
from PIL import Image, PngImagePlugin
from src.services.png_info.exceptions import InvalidImageError
from src.services.png_info.parser import parse_generation_parameters
from src.services.png_info.reader import read_info_from_image
from src.services.png_info.service import inspect_image_bytes
from src.services.png_info.writer import build_a1111_infotext, pnginfo_with_parameters

SAMPLE_GENINFO = """a cat in a hat
Negative prompt: blurry, low quality
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b"""


def _png_with_parameters(geninfo: str | None = None, extra: dict[str, str] | None = None) -> bytes:
    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    pnginfo = PngImagePlugin.PngInfo()
    if geninfo is not None:
        pnginfo.add_text("parameters", geninfo)
    if extra:
        for key, value in extra.items():
            pnginfo.add_text(key, value)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=pnginfo)
    return buffer.getvalue()


def test_read_info_from_image_extracts_parameters() -> None:
    image = Image.open(io.BytesIO(_png_with_parameters(SAMPLE_GENINFO, {"Description": "test image"})))
    geninfo, items = read_info_from_image(image)
    assert geninfo == SAMPLE_GENINFO
    assert items == {"Description": "test image"}


def test_read_info_from_image_without_metadata() -> None:
    image = Image.new("RGB", (8, 8), color="white")
    geninfo, items = read_info_from_image(image)
    assert geninfo is None
    assert items == {}


def test_parse_generation_parameters() -> None:
    parsed = parse_generation_parameters(SAMPLE_GENINFO)
    assert parsed["Prompt"] == "a cat in a hat"
    assert parsed["Negative prompt"] == "blurry, low quality"
    assert parsed["Steps"] == "20"
    assert parsed["Seed"] == "965400086"
    assert parsed["Size-1"] == 512
    assert parsed["Size-2"] == 512


def test_parse_generation_parameters_empty() -> None:
    assert parse_generation_parameters("") == {}
    assert parse_generation_parameters("   \n  ") == {}


def test_inspect_image_bytes() -> None:
    result = inspect_image_bytes(_png_with_parameters(SAMPLE_GENINFO))
    assert result.info == SAMPLE_GENINFO
    assert result.items["parameters"] == SAMPLE_GENINFO
    assert result.parameters["Prompt"] == "a cat in a hat"
    assert result.width == 64
    assert result.height == 64
    assert result.preview_base64 is not None
    assert result.preview_base64.startswith("data:image/jpeg;base64,")
    assert result.parameters["Sampler"] == "Euler a"


def test_inspect_image_bytes_without_metadata() -> None:
    image = Image.new("RGB", (32, 24), color="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = inspect_image_bytes(buffer.getvalue())
    assert result.info == ""
    assert result.items == {}
    assert result.parameters == {}
    assert result.width == 32
    assert result.height == 24


def test_inspect_image_bytes_rejects_invalid_payload() -> None:
    with pytest.raises(InvalidImageError):
        inspect_image_bytes(b"not-an-image")


def test_build_a1111_infotext_roundtrips_through_parser() -> None:
    infotext = build_a1111_infotext(
        prompt="a cat in a hat",
        negative_prompt="blurry, low quality",
        steps=30,
        sampler="euler_a",
        cfg_scale=7.5,
        seed=42,
        width=1024,
        height=1024,
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        lora_path=r"D:\loras\character.safetensors",
        lora_weight=0.8,
    )
    parsed = parse_generation_parameters(infotext)
    assert parsed["Prompt"] == "a cat in a hat"
    assert parsed["Negative prompt"] == "blurry, low quality"
    assert parsed["Steps"] == "30"
    assert parsed["Sampler"] == "Euler a"
    assert parsed["Seed"] == "42"
    assert parsed["Size-1"] == 1024
    assert parsed["Size-2"] == 1024
    assert parsed["Lora"] == "character"
    assert parsed["Lora weight"] == "0.8"


def test_build_a1111_infotext_formats_lora_stack() -> None:
    infotext = build_a1111_infotext(
        prompt="portrait",
        steps=20,
        sampler="euler",
        cfg_scale=7,
        seed=1,
        width=1024,
        height=1024,
        model_name="sdxl",
        loras=[
            (r"D:\loras\character.safetensors", 0.8),
            (r"D:\loras\style.safetensors", 0.4),
        ],
    )
    parsed = parse_generation_parameters(infotext)
    assert parsed["Lora"] == "character (0.8), style (0.4)"


def test_pnginfo_with_parameters_embeds_chunk() -> None:
    infotext = build_a1111_infotext(
        prompt="portrait",
        steps=20,
        sampler="euler",
        cfg_scale=7,
        seed=None,
        width=512,
        height=768,
        model_name="sdxl",
    )
    image = Image.new("RGB", (16, 16), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=pnginfo_with_parameters(infotext))
    result = inspect_image_bytes(buffer.getvalue())
    assert result.parameters["Prompt"] == "portrait"
    assert result.parameters["Sampler"] == "Euler"
    assert result.parameters["Seed"] == "-1"

