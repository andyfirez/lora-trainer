"""Worker wakeup: enqueue after commit, poll wait, and finish nudge."""

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.services.runnable.lifecycle import enqueue_runnable
from src.services.runnable.wakeup import nudge, reset_for_tests, wake_event
from src.services.worker.service import SubprocessRunnableWorker
from src.settings.app_settings import settings


@pytest.fixture(autouse=True)
def _reset_wake_event() -> Iterator[None]:
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.mark.asyncio
async def test_enqueue_runnable_nudges_after_commit(session: AsyncSession) -> None:
    sampling = Sampling(name="wake-me", config={"output_dir": "/tmp"})
    session.add(sampling)
    await session.flush()

    with patch("src.services.runnable.wakeup.nudge") as mock_nudge:
        await enqueue_runnable(
            session,
            sampling,
            kind="Sampling",
            entity_id=sampling.id,  # type: ignore[arg-type]
        )
        mock_nudge.assert_not_called()
        await session.commit()
        mock_nudge.assert_called_once()
    assert sampling.status == RunnableStatus.QUEUED


@pytest.mark.asyncio
async def test_wait_for_work_returns_before_poll_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.training, "worker_poll_interval_seconds", 5)
    worker = SubprocessRunnableWorker()
    task = asyncio.create_task(worker._wait_for_work())
    await asyncio.sleep(0)
    nudge()
    await asyncio.wait_for(task, timeout=1)
    assert not wake_event().is_set()


@pytest.mark.asyncio
async def test_run_entity_nudges_after_spawn_failure() -> None:
    worker = SubprocessRunnableWorker()
    handler = MagicMock()
    handler.build_command.side_effect = RuntimeError("boom")
    with (
        patch("src.services.worker.service.get_runnable_handler", return_value=handler),
        patch("src.services.worker.service.nudge") as mock_nudge,
        patch.object(worker, "_mark_spawn_failed", new=AsyncMock()),
    ):
        await worker._run_entity("sampling", 1)
    mock_nudge.assert_called()
