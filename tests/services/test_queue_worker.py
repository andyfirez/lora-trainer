import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.worker.service import QueueWorker


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self._index = 0

    def __iter__(self) -> "_FakeStdout":
        return self

    def __next__(self) -> bytes:
        if self._index >= len(self._lines):
            raise StopIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class _FakePopen:
    def __init__(self, lines: list[bytes]) -> None:
        self.stdout = _FakeStdout(lines)
        self.pid = 4242
        self._returncode: int | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode

    def wait(self) -> int:
        self._returncode = 0
        return 0


@pytest.mark.asyncio
async def test_queue_worker_start_stop() -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    with patch.object(worker, "_is_any_job_running", AsyncMock(return_value=False)), patch.object(
        worker, "_get_next_queued_job_id", AsyncMock(return_value=None)
    ):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()


@pytest.mark.asyncio
async def test_echo_subprocess_output_false_drains_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = QueueWorker(echo_subprocess_output=False)
    fake_proc = _FakePopen([b"secret line\n"])

    with caplog.at_level(logging.INFO, logger="src.services.worker.service"):
        with patch(
            "src.services.worker.service.subprocess.Popen",
            MagicMock(return_value=fake_proc),
        ), patch.object(worker, "_mark_job_running", AsyncMock()), patch.object(
            worker, "_finalize_job", AsyncMock()
        ):
            await worker._run_job(7)

    assert not any("[job 7]" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_echo_subprocess_output_true_logs_subprocess_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = QueueWorker(echo_subprocess_output=True)
    fake_proc = _FakePopen([b"visible line\n"])

    with caplog.at_level(logging.INFO, logger="src.services.worker.service"):
        with patch(
            "src.services.worker.service.subprocess.Popen",
            MagicMock(return_value=fake_proc),
        ), patch.object(worker, "_mark_job_running", AsyncMock()), patch.object(
            worker, "_finalize_job", AsyncMock()
        ):
            await worker._run_job(9)

    assert any("[job 9] visible line" in record.message for record in caplog.records)
