"""Sweep parameter models and helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

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

    def set_resolved_lora_paths(self, paths: list[str]) -> SweepParameters:
        if not paths:
            return self
        updated = self.model_copy(deep=True)
        updated.lora_path = SweepParameter(mode=SweepMode.VARY, values=paths)
        return updated


SourceType = Literal["manual", "training_job"]


class SweepCombination(BaseModel):
    index: int
    params: dict[str, Any]
