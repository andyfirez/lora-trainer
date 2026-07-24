"""Pick LoRA weight files from a work directory (final or checkpoint)."""

from __future__ import annotations

from pathlib import Path

from src.trainer.sdxl.checkpoint_state import (
    parse_epoch_from_checkpoint_name,
    parse_step_from_checkpoint_name,
)

_WEIGHT_EXTENSIONS = {".safetensors", ".ckpt"}


def is_checkpoint_weights(path: Path) -> bool:
    return (
        parse_epoch_from_checkpoint_name(path) is not None
        or parse_step_from_checkpoint_name(path) is not None
    )


def list_weight_files(work_dir: Path) -> list[Path]:
    if not work_dir.is_dir():
        return []
    return sorted(
        (
            child
            for child in work_dir.iterdir()
            if child.is_file()
            and child.suffix.lower() in _WEIGHT_EXTENSIONS
            and not child.name.startswith(".")
        ),
        key=lambda path: path.name.lower(),
    )


def pick_weights_file(work_dir: Path) -> Path | None:
    """Prefer final weights; fall back to the latest checkpoint by epoch/step."""
    candidates = list_weight_files(work_dir)
    if not candidates:
        return None

    finals = [path for path in candidates if not is_checkpoint_weights(path)]
    if finals:
        folder_matches = [path for path in finals if path.stem == work_dir.name]
        if folder_matches:
            return folder_matches[0]
        return max(finals, key=lambda path: path.stat().st_mtime)

    def checkpoint_sort_key(path: Path) -> tuple[int, int, float]:
        epoch = parse_epoch_from_checkpoint_name(path) or 0
        step = parse_step_from_checkpoint_name(path) or 0
        return (epoch, step, path.stat().st_mtime)

    return max(candidates, key=checkpoint_sort_key)


def work_dir_has_weights(work_dir: Path) -> bool:
    return pick_weights_file(work_dir) is not None
