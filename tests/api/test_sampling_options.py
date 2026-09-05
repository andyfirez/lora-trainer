"""API tests for sampling sampler/scheduler options."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_sampling_options_comfy_profile(client) -> None:
    response = await client.get("/samplings/options")
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "comfy"
    ids = [item["id"] for item in body["samplers"]]
    assert "euler" in ids
    assert "dpmpp_2m" in ids
    scheduler_ids = [item["id"] for item in body["schedulers"]]
    assert "karras" in scheduler_ids
    assert "automatic" not in scheduler_ids


@pytest.mark.asyncio
async def test_sampling_options_reforge_includes_automatic(client) -> None:
    response = await client.get("/samplings/options?profile=reforge")
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "reforge"
    assert body["schedulers"][0]["id"] == "automatic"
    assert body["schedulers"][0]["label"] == "Automatic"
    euler = next(item for item in body["samplers"] if item["id"] == "euler_ancestral")
    assert euler["label"] == "Euler a"
    assert euler["default_scheduler"] == "normal"
