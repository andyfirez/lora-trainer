"""Output directory resolution for sampling jobs."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sampler.config import SamplingConfig


def resolve_sampling_output_path(
    sampling_config: "SamplingConfig",
    job_id: int,
) -> Path:
    """Return the output directory for a sampling job.

    Jobs write to ``{output_dir}/job_{job_id}`` where ``output_dir`` is an
    absolute path configured by the user.
    """
    raw = sampling_config.output_dir.strip()
    if not raw:
        raise ValueError("output_dir is required")
    base = Path(raw).expanduser()
    if not base.is_absolute():
        raise ValueError("output_dir must be an absolute path")
    return base.resolve() / f"job_{job_id}"
