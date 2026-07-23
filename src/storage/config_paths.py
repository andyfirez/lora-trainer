"""Resolve managed storage paths for training and sampling configs."""

from pathlib import Path

from src.storage.paths import StoragePaths


def resolve_config_base_model(base_model_name: str) -> str:
    """Resolve a relative base model path to an absolute filesystem path."""
    if Path(base_model_name).is_absolute():
        return str(Path(base_model_name).expanduser().resolve())
    if "/" in base_model_name or "\\" in base_model_name or base_model_name.endswith((".safetensors", ".ckpt")):
        return str(StoragePaths.resolve_base_model(base_model_name))
    if StoragePaths.base_model_exists(base_model_name):
        return str(StoragePaths.resolve_base_model(base_model_name))
    raise ValueError(
        f"Base model must be a local path under base_models_root: {base_model_name}"
    )


def resolve_config_output_dir(output_dir: str) -> Path:
    if Path(output_dir).is_absolute():
        return Path(output_dir).expanduser().resolve()
    return StoragePaths.resolve_lora_path(output_dir)


def resolve_config_lora_file(lora_path: str) -> Path:
    path = Path(lora_path)
    if path.is_absolute():
        return path.expanduser().resolve()
    return StoragePaths.resolve_lora_path(lora_path)
