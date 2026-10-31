"""Resolve and validate local SDXL checkpoint files for training."""

from pathlib import Path

from src.storage.paths import StoragePaths
from src.trainer.sdxl.model_loader import is_checkpoint_file


def resolve_config_base_model(base_model_name: str) -> str:
    """Resolve to an absolute path of an existing .safetensors or .ckpt file."""
    raw = base_model_name.strip()
    if not raw:
        raise ValueError("base_model_name is required")

    if Path(raw).is_absolute():
        path = Path(raw).expanduser().resolve()
    else:
        path = StoragePaths.resolve_base_model(raw)

    if not is_checkpoint_file(path):
        raise ValueError(
            "Base model must be a local .safetensors or .ckpt checkpoint file "
            "(not a HuggingFace ID or model folder)."
        )
    if not path.is_file():
        raise ValueError(f"Base model file not found: {path}")
    return str(path)
