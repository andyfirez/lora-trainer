"""Output directory resolution for sampling runs."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sampler.config import SamplingConfig


def resolve_sampling_output_path(
    sampling_config: "SamplingConfig",
    sampling_id: int,
) -> Path:
    """Return the output directory for a sampling run.

    Runs write to ``{output_dir}/sampling_{sampling_id}`` where ``output_dir``
    is an absolute path configured by the user.
    """
    raw = sampling_config.output_dir.strip()
    if not raw:
        raise ValueError("output_dir is required")
    base = Path(raw).expanduser()
    if not base.is_absolute():
        raise ValueError("output_dir must be an absolute path")
    return base.resolve() / f"sampling_{sampling_id}"
