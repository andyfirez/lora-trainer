"""Sweep parameter models and helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SWEEP_PARAM_ORDER: tuple[str, ...] = (
    "lora_path",
    "lora_weight",
    "prompt",
    "cfg_scale",
    "steps",
    "seed",
    "base_model_name",
    "negative_prompt",
    "width",
    "height",
    "scheduler",
)


class SweepMode(str, Enum):
    FIXED = "fixed"
    VARY = "vary"


class LoraEntry(BaseModel):
    path: str | None = None
    trigger: str = ""


def normalize_lora_path_value(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_trigger_words(trigger: str) -> list[str]:
    return [word.strip() for word in trigger.split(",") if word.strip()]


def parse_lora_entry(value: Any) -> LoraEntry:
    if isinstance(value, LoraEntry):
        return value
    if isinstance(value, dict):
        path = value.get("path")
        trigger = value.get("trigger", "")
        return LoraEntry(
            path=normalize_lora_path_value(path),
            trigger=str(trigger or "").strip(),
        )
    if value is None or (isinstance(value, str) and not value.strip()):
        return LoraEntry(path=None, trigger="")
    if isinstance(value, str):
        return LoraEntry(path=normalize_lora_path_value(value), trigger="")
    return LoraEntry(path=normalize_lora_path_value(value), trigger="")


def lora_entry_path(entry: LoraEntry) -> str | None:
    return normalize_lora_path_value(entry.path)


def lora_entry_to_param_value(entry: LoraEntry) -> dict[str, Any]:
    return {"path": lora_entry_path(entry), "trigger": entry.trigger}


def format_lora_path_label(path: str | None, trigger: str = "") -> str:
    if path is None:
        return "base model"
    from pathlib import Path

    stem = Path(path).stem
    trigger = trigger.strip()
    if trigger:
        return f"{stem} ({trigger})"
    return stem


def dedupe_lora_entries(entries: list[LoraEntry]) -> list[LoraEntry]:
    """Collapse multiple empty LoRA entries into a single base-model cell."""
    result: list[LoraEntry] = []
    seen_empty = False
    seen_files: set[str] = set()
    for entry in entries:
        path = lora_entry_path(entry)
        if path is None:
            if not seen_empty:
                seen_empty = True
                result.append(LoraEntry(path=None, trigger=entry.trigger))
            continue
        if path not in seen_files:
            seen_files.add(path)
            result.append(LoraEntry(path=path, trigger=entry.trigger))
    return result


def lora_axis_display(param: SweepParameter) -> tuple[list[str | None], list[str]]:
    """Return axis keys (paths) and human-readable labels for grid headers."""
    entries = dedupe_lora_entries([parse_lora_entry(value) for value in param.effective_values()])
    if not entries:
        entries = [LoraEntry(path=None, trigger="")]
    paths = [lora_entry_path(entry) for entry in entries]
    labels = [format_lora_path_label(lora_entry_path(entry), entry.trigger) for entry in entries]
    return paths, labels


class SweepParameter(BaseModel):
    mode: SweepMode = SweepMode.FIXED
    value: Any = None
    values: list[Any] = Field(default_factory=list)

    def effective_values(self) -> list[Any]:
        if self.mode == SweepMode.VARY and self.values:
            return list(self.values)
        if self.value is not None:
            return [self.value]
        return []

    def is_varying(self) -> bool:
        return self.mode == SweepMode.VARY and len(self.values) > 1

    def first_value(self) -> Any:
        values = self.effective_values()
        if not values:
            return None
        return values[0]


class GridLayout(BaseModel):
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None


class SweepParameters(BaseModel):
    base_model_name: SweepParameter = Field(
        default_factory=lambda: SweepParameter(value="stabilityai/stable-diffusion-xl-base-1.0")
    )
    lora_path: SweepParameter = Field(default_factory=SweepParameter)
    lora_weight: SweepParameter = Field(default_factory=lambda: SweepParameter(value=1.0))
    prompt: SweepParameter = Field(default_factory=SweepParameter)
    negative_prompt: SweepParameter = Field(default_factory=lambda: SweepParameter(value=""))
    steps: SweepParameter = Field(default_factory=lambda: SweepParameter(value=30))
    cfg_scale: SweepParameter = Field(default_factory=lambda: SweepParameter(value=7.5))
    width: SweepParameter = Field(default_factory=lambda: SweepParameter(value=None))
    height: SweepParameter = Field(default_factory=lambda: SweepParameter(value=None))
    scheduler: SweepParameter = Field(default_factory=lambda: SweepParameter(value="euler"))
    seed: SweepParameter = Field(default_factory=lambda: SweepParameter(value=None))

    def vary_keys_with_values(self) -> list[str]:
        result: list[str] = []
        for key in SWEEP_PARAM_ORDER:
            param = getattr(self, key)
            if param.mode == SweepMode.VARY and len(param.values) > 0:
                result.append(key)
        return result

    def get_param(self, key: str) -> SweepParameter:
        return getattr(self, key)

    def set_resolved_lora_sweep_values(self, entries: list[LoraEntry]) -> SweepParameters:
        if not entries:
            return self
        payload = [lora_entry_to_param_value(entry) for entry in entries]
        updated = self.model_copy(deep=True)
        if len(payload) == 1:
            updated.lora_path = SweepParameter(mode=SweepMode.FIXED, value=payload[0])
        else:
            updated.lora_path = SweepParameter(mode=SweepMode.VARY, values=payload)
        return updated

    def set_resolved_lora_paths(self, paths: list[str]) -> SweepParameters:
        entries = [LoraEntry(path=path, trigger="") for path in paths]
        return self.set_resolved_lora_sweep_values(entries)


SourceType = Literal["manual", "training_job"]


class SweepCombination(BaseModel):
    index: int
    params: dict[str, Any]
