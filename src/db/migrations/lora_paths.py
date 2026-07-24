"""Helpers for normalizing trained LoRA path values."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def lora_root_from_config(config_path: Path | None = None) -> Path:
    path = config_path or Path(os.environ.get("APP_CONFIG_FILE", "config.toml"))
    if path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        storage = data.get("storage")
        if isinstance(storage, dict) and storage.get("lora_root"):
            return Path(str(storage["lora_root"])).expanduser().resolve()
    return Path("~/lora-trainer/lora").expanduser().resolve()


def normalize_lora_relative_path(relative_path: str, root: Path) -> str:
    """Convert an absolute path under root to a posix relative path; pass through otherwise."""
    path = Path(relative_path)
    if not path.is_absolute():
        return relative_path.strip().strip("/\\").replace("\\", "/")

    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return relative_path

    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return relative_path

    return "" if rel == Path(".") else rel.as_posix()


def to_lora_relative(root: Path, absolute_path: str | Path) -> str | None:
    try:
        resolved = Path(absolute_path).expanduser().resolve()
    except OSError:
        return None
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return "" if rel == Path(".") else rel.as_posix()
