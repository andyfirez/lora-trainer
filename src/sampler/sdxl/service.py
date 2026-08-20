"""Sweep sampling engine and job entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from src.sampler.config import SamplingConfig
from src.sampler.progress_callbacks import (
    ProgressCallback,
    ProgressStatusCallback,
)
from src.sampler.sweep.engine import SweepEngine
from src.trainer.concept_training_metadata import ConceptTrainingMetadata
from src.trainer.inference_config import SDXLInferenceConfig


def run_sweep_sampling(
    *,
    sampling_config: SamplingConfig,
    base_inference_config: SDXLInferenceConfig,
    output_dir: Path,
    sampling_id: int | None = None,
    progress_status_callback: ProgressStatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    log: logging.Logger | None = None,
    concept_metadata: dict[int, ConceptTrainingMetadata] | None = None,
    compose_grids: bool = True,
) -> None:
    """Run a parameter sweep sampling job."""
    engine = SweepEngine(
        sampling_config,
        base_inference_config=base_inference_config,
        output_dir=output_dir,
        sampling_id=sampling_id,
        progress_status_callback=progress_status_callback,
        progress_callback=progress_callback,
        log=log,
        concept_metadata=concept_metadata,
        compose_grids=compose_grids,
    )
    engine.run()
