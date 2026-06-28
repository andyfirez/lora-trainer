"""Output directory resolution for sampling jobs."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.sampler.config import SamplingConfig
    from src.trainer.config import TrainConfig


def resolve_sampling_output_path(
    sampling_config: "SamplingConfig",
    job_id: int,
    source_train_config: Optional["TrainConfig"] = None,
) -> Path:
    """Return the output directory for a sampling job.

    Train-linked jobs write to ``{train.output_dir}/{train.lora_name}/samples``.
    Standalone jobs write to ``{sampling.output_dir}/samples/job_{job_id}``.
    """
    if source_train_config is not None:
        return Path(source_train_config.output_dir) / source_train_config.lora_name / "samples"
    return Path(sampling_config.output_dir) / "samples" / f"job_{job_id}"
