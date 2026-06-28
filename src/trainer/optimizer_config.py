"""Optimizer configuration, presets, and factory."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PRESETS_PATH = (
    Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "optimizer_presets.json"
)


class Optimizer(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    ADAFACTOR = "adafactor"
    PRODIGY = "prodigy"


class OptimizerConfig(BaseModel):
    type: Optimizer = Optimizer.ADAMW_8BIT
    weight_decay: float = Field(default=0.01, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    relative_step: bool = False
    scale_parameter: bool = False
    warmup_init: bool = False
    decouple: bool = True
    use_bias_correction: bool = True
    safeguard_warmup: bool = True
    d0: float = Field(default=1e-5, gt=0.0)
    d_coef: float = Field(default=1.0, gt=0.0)

    @classmethod
    def defaults(cls) -> OptimizerConfig:
        return get_optimizer_preset(Optimizer.ADAMW_8BIT).optimizer


class OptimizerPreset(BaseModel):
    optimizer: OptimizerConfig
    learning_rate: float = Field(gt=0.0)
    lr_scheduler: str
    lr_warmup_steps: int = Field(ge=0)


def load_optimizer_presets() -> dict[str, Any]:
    return json.loads(_PRESETS_PATH.read_text(encoding="utf-8"))


def get_optimizer_preset(optimizer_type: Optimizer) -> OptimizerPreset:
    presets = load_optimizer_presets()
    raw = presets[optimizer_type.value]
    return OptimizerPreset.model_validate(raw)


def build_optimizer(params: list[Any], config: Any) -> Any:
    import torch

    from src.trainer.config import TrainConfig

    assert isinstance(config, TrainConfig)
    lr = config.learning_rate
    opt = config.optimizer
    betas = (opt.beta1, opt.beta2)

    if opt.type == Optimizer.ADAMW:
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=opt.weight_decay)
    if opt.type == Optimizer.ADAMW_8BIT:
        from bitsandbytes.optim import AdamW8bit

        return AdamW8bit(params, lr=lr, betas=betas, weight_decay=opt.weight_decay)
    if opt.type == Optimizer.ADAFACTOR:
        from transformers.optimization import Adafactor

        return Adafactor(
            params,
            lr=lr,
            relative_step=opt.relative_step,
            scale_parameter=opt.scale_parameter,
            warmup_init=opt.warmup_init,
        )
    if opt.type == Optimizer.PRODIGY:
        from prodigyopt import Prodigy

        return Prodigy(
            params,
            lr=lr,
            betas=betas,
            weight_decay=opt.weight_decay,
            decouple=opt.decouple,
            use_bias_correction=opt.use_bias_correction,
            safeguard_warmup=opt.safeguard_warmup,
            d0=opt.d0,
            d_coef=opt.d_coef,
        )
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=opt.weight_decay)
