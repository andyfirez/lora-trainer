"""Create sampling optionally enqueues in the same request."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.routers.samplings import _create_sampling
from src.api.schemas.samplings import CreateSamplingRequest


@pytest.mark.asyncio
async def test_create_sampling_enqueues_when_requested() -> None:
    service = AsyncMock()
    draft = MagicMock(id=7)
    queued = MagicMock(id=7, status="queued")
    service.create_sampling = AsyncMock(return_value=draft)
    service.enqueue_sampling = AsyncMock(return_value=queued)

    result = await _create_sampling(
        service,
        CreateSamplingRequest(name="Playground", config={"prompt": "x"}, enqueue=True),
    )

    service.create_sampling.assert_awaited_once()
    service.enqueue_sampling.assert_awaited_once_with(7)
    assert result is queued


@pytest.mark.asyncio
async def test_create_sampling_skips_enqueue_by_default() -> None:
    service = AsyncMock()
    draft = MagicMock(id=3)
    service.create_sampling = AsyncMock(return_value=draft)
    service.enqueue_sampling = AsyncMock()

    result = await _create_sampling(
        service,
        CreateSamplingRequest(name="Draft", config={"prompt": "x"}),
    )

    service.enqueue_sampling.assert_not_awaited()
    assert result is draft
