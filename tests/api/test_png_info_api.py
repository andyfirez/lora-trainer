"""API tests for PNG Info endpoint."""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image, PngImagePlugin

from src.api.main import app

SAMPLE_GENINFO = """a cat in a hat
Negative prompt: blurry
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 512x512"""


def _png_bytes(geninfo: str | None = None) -> bytes:
    image = Image.new("RGB", (48, 48), color=(10, 20, 30))
    buffer = io.BytesIO()
    if geninfo is None:
        image.save(buffer, format="PNG")
    else:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", geninfo)
        image.save(buffer, format="PNG", pnginfo=pnginfo)
    return buffer.getvalue()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_png_info_returns_metadata(client) -> None:
    response = await client.post(
        "/png-info",
        files={"file": ("sample.png", _png_bytes(SAMPLE_GENINFO), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["info"] == SAMPLE_GENINFO
    assert body["items"]["parameters"] == SAMPLE_GENINFO
    assert body["parameters"]["Prompt"] == "a cat in a hat"
    assert body["parameters"]["Seed"] == "123"
    assert body["width"] == 48
    assert body["height"] == 48
    assert body["preview_base64"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_png_info_empty_file(client) -> None:
    response = await client.post(
        "/png-info",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file"


@pytest.mark.asyncio
async def test_png_info_invalid_file(client) -> None:
    response = await client.post(
        "/png-info",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_png_info_no_metadata(client) -> None:
    response = await client.post(
        "/png-info",
        files={"file": ("plain.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["info"] == ""
    assert body["items"] == {}
    assert body["parameters"] == {}
