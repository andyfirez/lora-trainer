"""Output directory resolution for sampling runs."""

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sampler.config import SamplingConfig

DEFAULT_SAMPLING_OUTPUT_DIR = "output"


def effective_sampling_output_dir(sampling_config: "SamplingConfig") -> str:
    raw = sampling_config.output_dir.strip()
    return raw or DEFAULT_SAMPLING_OUTPUT_DIR


def _sampling_project_root() -> Path:
    """Root directory for relative sampling output paths (same convention as logs/)."""
    return Path.cwd().resolve()


def resolve_sampling_config_output_dir(output_dir: str) -> Path:
    """Resolve sampling output_dir relative to the app working directory.

    Training LoRA artifacts use ``lora_root`` for relative paths; playground
    sampling instead writes under the project folder (e.g. ``./output``).
    Absolute paths are accepted as-is.
    """
    raw = output_dir.strip() or DEFAULT_SAMPLING_OUTPUT_DIR
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()

    root = _sampling_project_root()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output_dir escapes project directory: {raw}") from exc
    return resolved


def resolve_sampling_output_path(
    sampling_config: "SamplingConfig",
    sampling_id: int | None = None,
) -> Path:
    """Return the output directory for a sampling run.

    Runs write PNGs to ``{output_dir}/{YYYY-MM-DD}/`` where ``output_dir``
    defaults to ``output`` when unset and may be absolute or relative to the
    application working directory.
    """
    _ = sampling_id
    base = resolve_sampling_config_output_dir(effective_sampling_output_dir(sampling_config))
    return base / date.today().isoformat()


def flat_grid_filename(sampling_id: int | None, index: int, title: str = "") -> str:
    suffix = ""
    if title:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)[:60]
        suffix = f"_{safe}"
    if sampling_id is not None:
        return f"{sampling_id}_grid_{index:03d}{suffix}.png"
    return f"grid_{index:03d}{suffix}.png"


def flat_sample_filename(sampling_id: int | None, seed: int | None, index: int) -> str:
    """Build a flat PNG filename unique within a shared date folder."""
    if seed is not None:
        if sampling_id is not None:
            return f"{sampling_id}_{int(seed)}.png"
        return f"{int(seed)}.png"
    if sampling_id is not None:
        return f"{sampling_id}_{index:04d}.png"
    return f"cell_{index:04d}.png"
