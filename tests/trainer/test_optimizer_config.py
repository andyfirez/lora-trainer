"""Tests for optimizer config, presets, and factory."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.trainer.config import LRScheduler, TrainConfig
from src.trainer.optimizer_config import (
    Optimizer,
    OptimizerConfig,
    build_optimizer,
    get_optimizer_preset,
    load_optimizer_presets,
)


class _ParamModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


@pytest.fixture
def trainable_params() -> list[nn.Parameter]:
    return list(_ParamModule().parameters())


def test_load_optimizer_presets_has_all_types() -> None:
    presets = load_optimizer_presets()
    assert set(presets.keys()) == {"adamw", "adamw_8bit", "adafactor", "prodigy"}


@pytest.mark.parametrize(
    ("optimizer_type", "expected_lr", "expected_scheduler", "expected_warmup"),
    [
        (Optimizer.ADAMW, 5e-5, "constant", 0),
        (Optimizer.ADAMW_8BIT, 5e-5, "constant", 0),
        (Optimizer.ADAFACTOR, 1e-4, "constant_with_warmup", 10),
        (Optimizer.PRODIGY, 1.0, "constant", 0),
    ],
)
def test_get_optimizer_preset_training_params(
    optimizer_type: Optimizer,
    expected_lr: float,
    expected_scheduler: str,
    expected_warmup: int,
) -> None:
    preset = get_optimizer_preset(optimizer_type)
    assert preset.optimizer.type == optimizer_type
    assert preset.learning_rate == expected_lr
    assert preset.lr_scheduler == expected_scheduler
    assert preset.lr_warmup_steps == expected_warmup


def test_get_optimizer_preset_prodigy_params() -> None:
    preset = get_optimizer_preset(Optimizer.PRODIGY)
    opt = preset.optimizer
    assert opt.weight_decay == 0.01
    assert opt.beta1 == 0.9
    assert opt.beta2 == 0.99
    assert opt.decouple is True
    assert opt.use_bias_correction is True
    assert opt.safeguard_warmup is True
    assert opt.d0 == pytest.approx(1e-5)
    assert opt.d_coef == 1.0


def test_get_optimizer_preset_adafactor_params() -> None:
    preset = get_optimizer_preset(Optimizer.ADAFACTOR)
    opt = preset.optimizer
    assert opt.relative_step is False
    assert opt.scale_parameter is False
    assert opt.warmup_init is False


def test_optimizer_config_defaults_match_adamw_8bit_preset() -> None:
    defaults = OptimizerConfig.defaults()
    preset = get_optimizer_preset(Optimizer.ADAMW_8BIT)
    assert defaults.model_dump() == preset.optimizer.model_dump()


def test_train_config_yaml_roundtrip_nested_optimizer() -> None:
    config = TrainConfig(
        optimizer=OptimizerConfig(
            type=Optimizer.PRODIGY,
            weight_decay=0.02,
            beta1=0.85,
            beta2=0.95,
            decouple=False,
        ),
        learning_rate=1.0,
        lr_scheduler=LRScheduler.CONSTANT,
    )
    restored = TrainConfig.from_yaml(config.to_yaml())
    assert restored.optimizer.type == Optimizer.PRODIGY
    assert restored.optimizer.weight_decay == 0.02
    assert restored.optimizer.beta1 == 0.85
    assert restored.optimizer.decouple is False
    assert restored.learning_rate == 1.0


def test_build_optimizer_adamw_uses_config_params(trainable_params: list[nn.Parameter]) -> None:
    config = TrainConfig(
        learning_rate=2e-4,
        optimizer=OptimizerConfig(
            type=Optimizer.ADAMW,
            weight_decay=0.05,
            beta1=0.8,
            beta2=0.88,
        ),
    )
    optimizer = build_optimizer(trainable_params, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.88)


@patch("bitsandbytes.optim.AdamW8bit")
def test_build_optimizer_adamw_8bit_uses_config_params(
    mock_adamw8bit: MagicMock,
    trainable_params: list[nn.Parameter],
) -> None:
    mock_adamw8bit.return_value = MagicMock()
    config = TrainConfig(
        learning_rate=3e-4,
        optimizer=OptimizerConfig(type=Optimizer.ADAMW_8BIT, weight_decay=0.03),
    )
    build_optimizer(trainable_params, config)
    mock_adamw8bit.assert_called_once_with(
        trainable_params,
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=0.03,
    )


@patch("transformers.optimization.Adafactor")
def test_build_optimizer_adafactor_uses_config_params(
    mock_adafactor: MagicMock,
    trainable_params: list[nn.Parameter],
) -> None:
    mock_adafactor.return_value = MagicMock()
    config = TrainConfig(
        learning_rate=5e-4,
        optimizer=OptimizerConfig(
            type=Optimizer.ADAFACTOR,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True,
        ),
    )
    build_optimizer(trainable_params, config)
    mock_adafactor.assert_called_once_with(
        trainable_params,
        lr=5e-4,
        relative_step=True,
        scale_parameter=True,
        warmup_init=True,
    )


@patch("prodigyopt.Prodigy")
def test_build_optimizer_prodigy_uses_config_params(
    mock_prodigy: MagicMock,
    trainable_params: list[nn.Parameter],
) -> None:
    mock_prodigy.return_value = MagicMock()
    config = TrainConfig(
        learning_rate=1.0,
        optimizer=OptimizerConfig(
            type=Optimizer.PRODIGY,
            weight_decay=0.02,
            beta1=0.91,
            beta2=0.98,
            decouple=False,
            use_bias_correction=False,
            safeguard_warmup=False,
            d0=2e-5,
            d_coef=2.0,
        ),
    )
    build_optimizer(trainable_params, config)
    mock_prodigy.assert_called_once_with(
        trainable_params,
        lr=1.0,
        betas=(0.91, 0.98),
        weight_decay=0.02,
        decouple=False,
        use_bias_correction=False,
        safeguard_warmup=False,
        d0=2e-5,
        d_coef=2.0,
    )
