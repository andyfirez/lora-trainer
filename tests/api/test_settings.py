import os

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.settings.app_settings import settings


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("APP_CONFIG_FILE", str(path))
    return path


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_get_settings_returns_worker_and_system_fields(client, config_file) -> None:
    config_file.write_text(
        """[server]
host = "127.0.0.1"
port = 8000

[database]
path = "test.db"
echo = false

[training]
worker_poll_interval_seconds = 5
max_concurrent_jobs = 1
logs_dir = "logs"
cancel_poll_interval_seconds = 1
""",
        encoding="utf-8",
    )

    response = await client.get("/settings/")
    assert response.status_code == 200
    body = response.json()
    assert body["max_concurrent_jobs"] == settings.training.max_concurrent_jobs
    assert body["worker_poll_interval_seconds"] == settings.training.worker_poll_interval_seconds
    assert body["server"]["host"] == settings.server.host
    assert body["database"]["path"] == settings.database.path
    assert body["training"]["logs_dir"] == settings.training.logs_dir
    assert body["config_file"] == str(config_file.resolve())
    assert body["app_version"] == "0.1.0"
    assert "cuda_available" in body["gpu"]


@pytest.mark.asyncio
async def test_patch_settings_updates_runtime_and_persists(client, config_file, monkeypatch) -> None:
    monkeypatch.setattr(settings.training, "max_concurrent_jobs", 1)
    monkeypatch.setattr(settings.training, "worker_poll_interval_seconds", 5)

    response = await client.patch(
        "/settings/",
        json={"max_concurrent_jobs": 3, "worker_poll_interval_seconds": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["max_concurrent_jobs"] == 3
    assert body["worker_poll_interval_seconds"] == 10
    assert settings.training.max_concurrent_jobs == 3
    assert settings.training.worker_poll_interval_seconds == 10

    saved = config_file.read_text(encoding="utf-8")
    assert "max_concurrent_jobs = 3" in saved
    assert "worker_poll_interval_seconds = 10" in saved


@pytest.mark.asyncio
async def test_patch_settings_requires_at_least_one_field(client) -> None:
    response = await client.patch("/settings/", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_settings_validates_minimum_values(client) -> None:
    response = await client.patch("/settings/", json={"max_concurrent_jobs": 0})
    assert response.status_code == 422

    response = await client.patch("/settings/", json={"worker_poll_interval_seconds": 0})
    assert response.status_code == 422
