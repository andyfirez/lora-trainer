"""Tests for SubprocessRunnableWorker spawn-failure status handling."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from src.db.tables.lora import Lora
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.worker.service import SubprocessRunnableWorker


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, entity: object) -> None:
        self.added.append(entity)

    async def commit(self) -> None:
        return None


@asynccontextmanager
async def _fake_session_factory(session: _FakeSession):
    yield session


async def _mark_spawn_failed(entity: Lora, error_message: str = "spawn failed") -> _FakeSession:
    session = _FakeSession()
    worker = SubprocessRunnableWorker()
    with (
        patch("src.services.worker.service.session_factory", lambda: _fake_session_factory(session)),
        patch("src.services.worker.service.get_by_kind", AsyncMock(return_value=entity)),
    ):
        await worker._mark_spawn_failed("lora", 1, error_message)
    return session


@pytest.mark.asyncio
async def test_mark_spawn_failed_marks_queued_entity_failed() -> None:
    entity = Lora(name="queued", config_yaml="base_model_name: x", status=RunnableStatus.QUEUED)
    session = await _mark_spawn_failed(entity, "boom")
    assert entity.status == RunnableStatus.FAILED
    assert entity.error_message == "boom"
    assert session.added == [entity]


@pytest.mark.asyncio
async def test_mark_spawn_failed_does_not_overwrite_running() -> None:
    entity = Lora(name="running", config_yaml="base_model_name: x", status=RunnableStatus.RUNNING)
    session = await _mark_spawn_failed(entity)
    assert entity.status == RunnableStatus.RUNNING
    assert session.added == []


@pytest.mark.asyncio
async def test_mark_spawn_failed_does_not_overwrite_cancelled() -> None:
    entity = Lora(name="cancelled", config_yaml="base_model_name: x", status=RunnableStatus.CANCELLED)
    session = await _mark_spawn_failed(entity)
    assert entity.status == RunnableStatus.CANCELLED
    assert session.added == []
