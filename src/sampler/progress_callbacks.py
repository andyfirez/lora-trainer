"""Optional progress/status callbacks shared by sampler and sweep engine."""

from __future__ import annotations

from typing import Callable

ProgressStatusCallback = Callable[[str | None], None]
ProgressCallback = Callable[[int, int], None]


class ProgressCallbackMixin:
    _progress_status_callback: ProgressStatusCallback | None
    _progress_callback: ProgressCallback | None

    def _set_status(self, status: str | None) -> None:
        if self._progress_status_callback is not None:
            self._progress_status_callback(status)

    def _set_progress(self, step: int, total: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(step, total)
