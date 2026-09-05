"""API tests for sampling sampler/scheduler options."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.main import app
from src.trainer.sdxl.prompt_chunking import PromptChunkCounts


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


@pytest.mark.asyncio
@patch("src.api.routers.samplings.count_prompt_chunks", return_value=PromptChunkCounts(tokens=142, chunks=2))
@patch("src.api.routers.samplings.load_sdxl_tokenizers", return_value=(MagicMock(), MagicMock()))
async def test_token_count_returns_max_window_stats(
    _load: object,
    _count: object,
    client,
) -> None:
    response = await client.post(
        "/samplings/token-count",
        json={"prompt": "a very long prompt", "base_model_name": "local-model.safetensors"},
    )
    assert response.status_code == 200
    assert response.json() == {"tokens": 142, "chunks": 2, "window": 75}


@pytest.mark.asyncio
@patch("src.api.routers.samplings.load_sdxl_tokenizers", side_effect=ValueError("missing model"))
async def test_token_count_invalid_model_returns_400(_load: object, client) -> None:
    response = await client.post(
        "/samplings/token-count",
        json={"prompt": "cat", "base_model_name": "missing"},
    )
    assert response.status_code == 400
    assert "missing model" in response.json()["detail"]
