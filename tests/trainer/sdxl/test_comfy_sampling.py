import pytest
import torch
from diffusers import DDPMScheduler
from pydantic import ValidationError

from src.sampler.config import SamplingConfig
from src.trainer.sdxl.latent_sampling.comfy.constants import validate_sampler_scheduler_pair
from src.trainer.sdxl.latent_sampling.comfy.model_sampling import EpsModelSampling
from src.trainer.sdxl.latent_sampling.comfy.noise import prepare_noise
from src.trainer.sdxl.latent_sampling.comfy.plan import build_comfy_sampling_plan
from src.trainer.sdxl.latent_sampling.comfy.schedulers import calculate_sigmas, get_sigmas_karras, simple_scheduler
from src.trainer.sdxl.latent_sampling.comfy.samplers import SAMPLER_FUNCTIONS


def _model_sampling() -> EpsModelSampling:
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    return EpsModelSampling(scheduler.alphas_cumprod)


def test_simple_scheduler_has_terminal_zero() -> None:
    model_sampling = _model_sampling()
    sigmas = simple_scheduler(model_sampling, steps=20)
    assert len(sigmas) == 21
    assert sigmas[-1] == 0.0
    assert sigmas[0] > sigmas[1] > sigmas[-2]


def test_karras_scheduler_has_terminal_zero() -> None:
    model_sampling = _model_sampling()
    sigmas = get_sigmas_karras(20, model_sampling.sigma_min, model_sampling.sigma_max)
    assert len(sigmas) == 21
    assert sigmas[-1] == 0.0


def test_prepare_noise_is_deterministic_on_cpu() -> None:
    shape = (1, 4, 8, 8)
    first = prepare_noise(shape, seed=123, dtype=torch.float32, device=torch.device("cpu"))
    second = prepare_noise(shape, seed=123, dtype=torch.float32, device=torch.device("cpu"))
    assert torch.equal(first, second)


def test_validate_sampler_scheduler_pair_rejects_legacy_combo() -> None:
    with pytest.raises(ValueError, match="Unsupported sampler/scheduler pair"):
        validate_sampler_scheduler_pair("dpmpp_2m", "simple")


def test_sampling_config_rejects_legacy_unified_scheduler_yaml() -> None:
    yaml_text = """
output_dir: output
parameters:
  prompt:
    mode: fixed
    value: test
  scheduler:
    mode: fixed
    value: euler
"""
    with pytest.raises(ValidationError, match="Legacy unified parameters.scheduler"):
        SamplingConfig.from_yaml(yaml_text)


def test_build_comfy_sampling_plan_supports_v1_pairs() -> None:
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    for sampler_name, scheduler_name in (
        ("euler", "simple"),
        ("euler_ancestral", "simple"),
        ("dpmpp_2m", "karras"),
    ):
        plan = build_comfy_sampling_plan(
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            steps=5,
            noise_scheduler=scheduler,
            device=torch.device("cpu"),
        )
        assert plan.sampler_name == sampler_name
        assert plan.scheduler_name == scheduler_name
        assert len(plan.sigmas) == 6
        assert sampler_name in SAMPLER_FUNCTIONS


def test_calculate_sigmas_dispatches_named_schedulers() -> None:
    model_sampling = _model_sampling()
    simple = calculate_sigmas(model_sampling, "simple", 10)
    karras = calculate_sigmas(model_sampling, "karras", 10)
    assert len(simple) == 11
    assert len(karras) == 11
