"""Checkpoint + resume-state helpers for SDXL LoRA training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_EPOCH_RE = re.compile(r"_epoch(?P<epoch>\d+)\.")
_STEP_RE = re.compile(r"_step(?P<step>\d+)\.")


@dataclass(frozen=True)
class ResumeState:
    checkpoint_path: Path
    state_path: Path
    epoch: int
    global_step: int
    epoch_step: int
    lora_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    lr_scheduler_state_dict: dict[str, Any]
    grad_scaler_state_dict: dict[str, Any] | None


def state_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.state.pt")


def find_latest_checkpoint(work_dir: Path, lora_name: str, output_ext: str) -> Path | None:
    pattern_epoch = f"{lora_name}_epoch*.{output_ext}"
    pattern_step = f"{lora_name}_step*.{output_ext}"
    candidates = list(work_dir.glob(pattern_epoch)) + list(work_dir.glob(pattern_step))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_epoch_from_checkpoint_name(path: Path) -> int | None:
    match = _EPOCH_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("epoch"))


def parse_step_from_checkpoint_name(path: Path) -> int | None:
    match = _STEP_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("step"))


def prune_stale_resume_states(work_dir: Path, *, keep_state_path: Path) -> None:
    """Delete resume state files except the one for the latest checkpoint."""
    keep_resolved = keep_state_path.resolve()
    for state_path in work_dir.glob("*.state.pt"):
        if state_path.resolve() != keep_resolved:
            state_path.unlink(missing_ok=True)


def delete_all_resume_states(work_dir: Path) -> None:
    """Remove all resume state files from a training work directory."""
    for state_path in work_dir.glob("*.state.pt"):
        state_path.unlink(missing_ok=True)


def save_resume_state(
    *,
    checkpoint_path: Path,
    lora_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    lr_scheduler_state_dict: dict[str, Any],
    epoch: int,
    global_step: int,
    epoch_step: int,
    grad_scaler_state_dict: dict[str, Any] | None = None,
) -> Path:
    state_path = state_path_for_checkpoint(checkpoint_path)
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "epoch_step": int(epoch_step),
        "lora_state_dict": lora_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
    }
    if grad_scaler_state_dict is not None:
        payload["grad_scaler_state_dict"] = grad_scaler_state_dict
    torch.save(payload, state_path)
    return state_path


def load_resume_state(checkpoint_path: Path) -> ResumeState:
    state_path = state_path_for_checkpoint(checkpoint_path)
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume state file is missing for checkpoint: {checkpoint_path}")
    raw = torch.load(state_path, map_location="cpu")
    scaler_raw = raw.get("grad_scaler_state_dict")
    grad_scaler_state_dict = dict(scaler_raw) if scaler_raw is not None else None
    return ResumeState(
        checkpoint_path=checkpoint_path,
        state_path=state_path,
        epoch=int(raw.get("epoch", 0)),
        global_step=int(raw.get("global_step", 0)),
        epoch_step=int(raw.get("epoch_step", 0)),
        lora_state_dict=dict(raw.get("lora_state_dict", {})),
        optimizer_state_dict=dict(raw.get("optimizer_state_dict", {})),
        lr_scheduler_state_dict=dict(raw.get("lr_scheduler_state_dict", {})),
        grad_scaler_state_dict=grad_scaler_state_dict,
    )
