"""Shared types for vendored Comfy inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComfyInferenceStack:
    model: Any
    clip: Any
    vae: Any
    base_model_path: str
    lora_state: dict[str, Any] | None = None
    lora_apply_te1: bool = False
    lora_apply_te2: bool = False
    _base_model: Any = field(repr=False, default=None)
    _base_clip: Any = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if self._base_model is None:
            self._base_model = self.model
        if self._base_clip is None:
            self._base_clip = self.clip
