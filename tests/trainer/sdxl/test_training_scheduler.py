"""Tests for train forward-process noise scheduler construction."""

import torch
from diffusers import DDPMScheduler, EulerDiscreteScheduler
from src.trainer.sdxl.model_loader import _build_training_noise_scheduler

_SCHEDULER_KWARGS = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
}


def test_build_training_noise_scheduler_from_euler_returns_ddpm() -> None:
    source = EulerDiscreteScheduler(**_SCHEDULER_KWARGS)
    scheduler = _build_training_noise_scheduler(source)
    assert isinstance(scheduler, DDPMScheduler)


def test_build_training_noise_scheduler_preserves_betas_and_prediction_type() -> None:
    source = EulerDiscreteScheduler(**_SCHEDULER_KWARGS)
    scheduler = _build_training_noise_scheduler(source)
    assert scheduler.config.num_train_timesteps == source.config.num_train_timesteps
    assert scheduler.config.prediction_type == source.config.prediction_type
    assert scheduler.config.beta_start == source.config.beta_start
    assert scheduler.config.beta_end == source.config.beta_end
    assert scheduler.config.beta_schedule == source.config.beta_schedule


def test_build_training_noise_scheduler_add_noise_is_variance_preserving() -> None:
    source = EulerDiscreteScheduler(**_SCHEDULER_KWARGS)
    euler = source
    ddpm = _build_training_noise_scheduler(source)

    torch.manual_seed(0)
    x = torch.randn(1, 4, 128, 128)
    noise = torch.randn_like(x)

    t_low = torch.tensor([0])
    t_high = torch.tensor([999])
    ddpm_low = float(torch.linalg.norm(ddpm.add_noise(x, noise, t_low)))
    ddpm_high = float(torch.linalg.norm(ddpm.add_noise(x, noise, t_high)))
    euler_low = float(torch.linalg.norm(euler.add_noise(x, noise, t_low)))
    euler_high = float(torch.linalg.norm(euler.add_noise(x, noise, t_high)))

    assert ddpm_high / ddpm_low < 1.5
    assert euler_high / euler_low > 10.0
