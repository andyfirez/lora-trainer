"""Human-readable sampling progress phase labels."""

from __future__ import annotations

PHASE_STARTING = "Starting"
PHASE_LOADING_MODEL = "Loading model"
PHASE_GENERATING = "Generating"
PHASE_VAE_DECODING = "VAE decoding"
PHASE_SAVING = "Saving"
PHASE_COMPOSING_GRIDS = "Composing grids"

_SEPARATOR = " · "


def format_sampling_progress_title(
    phase: str,
    *,
    cell_index: int | None = None,
    cell_total: int | None = None,
    step: int | None = None,
    step_total: int | None = None,
) -> str:
    parts = [phase]
    if cell_index is not None and cell_total is not None:
        parts.append(f"Cell {cell_index}/{cell_total}")
    if phase == PHASE_GENERATING and step is not None and step_total is not None:
        parts.append(f"{step}/{step_total}")
    return _SEPARATOR.join(parts)
