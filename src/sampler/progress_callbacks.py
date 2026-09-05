"""Optional progress/status callbacks shared by sampler and sweep engine."""

from __future__ import annotations

from typing import Callable

from src.sampler.progress_labels import (
    PHASE_GENERATING,
    PHASE_SAVING,
    PHASE_VAE_DECODING,
    format_sampling_progress_title,
)

ProgressStatusCallback = Callable[[str | None], None]
ProgressCallback = Callable[[int, int], None]
ProgressPhaseCallback = Callable[[str], None]

_CELL_CONTEXT_PHASES = frozenset({PHASE_GENERATING, PHASE_VAE_DECODING, PHASE_SAVING})


class ProgressCallbackMixin:
    _progress_status_callback: ProgressStatusCallback | None
    _progress_callback: ProgressCallback | None

    def _set_status(self, status: str | None) -> None:
        if self._progress_status_callback is not None:
            self._progress_status_callback(status)

    def _set_progress(self, step: int, total: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(step, total)


class SamplingProgressReporter(ProgressCallbackMixin):
    def __init__(
        self,
        progress_status_callback: ProgressStatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._progress_status_callback = progress_status_callback
        self._progress_callback = progress_callback
        self._phase: str | None = None
        self._cell_index: int | None = None
        self._cell_total: int | None = None
        self._cell_step: int | None = None
        self._cell_step_total: int | None = None

    def set_phase(
        self,
        phase: str,
        *,
        cell_index: int | None = None,
        cell_total: int | None = None,
    ) -> None:
        self._phase = phase
        if cell_index is not None:
            self._cell_index = cell_index
        if cell_total is not None:
            self._cell_total = cell_total
        if phase not in _CELL_CONTEXT_PHASES:
            if cell_index is None:
                self._cell_index = None
            if cell_total is None:
                self._cell_total = None
            self._cell_step = None
            self._cell_step_total = None
        elif phase != PHASE_GENERATING:
            self._cell_step = None
            self._cell_step_total = None
        self._sync_status()

    def set_progress(
        self,
        step: int,
        total: int,
        *,
        cell_step: int | None = None,
        cell_step_total: int | None = None,
    ) -> None:
        if cell_step is not None:
            self._cell_step = cell_step
        if cell_step_total is not None:
            self._cell_step_total = cell_step_total
        self._set_progress(step, total)
        self._sync_status()

    def clear(self) -> None:
        self._phase = None
        self._cell_index = None
        self._cell_total = None
        self._cell_step = None
        self._cell_step_total = None
        self._set_status(None)

    def _sync_status(self) -> None:
        if self._phase is None:
            self._set_status(None)
            return
        include_steps = self._phase == PHASE_GENERATING
        self._set_status(
            format_sampling_progress_title(
                self._phase,
                cell_index=self._cell_index,
                cell_total=self._cell_total,
                step=self._cell_step if include_steps else None,
                step_total=self._cell_step_total if include_steps else None,
            )
        )
