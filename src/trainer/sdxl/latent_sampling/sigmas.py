"""Sigma schedules ported from ComfyUI KSampler.SCHEDULER_HANDLERS."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy
import torch
from torch import Tensor

from src.sampler.registry.samplers import DISCARD_PENULTIMATE_SIGMA_SAMPLERS
from src.trainer.sdxl.latent_sampling.model_sampling import DiscreteEpsSampling


class SchedulerHandler(NamedTuple):
    handler: Callable[..., Tensor]
    use_ms: bool = True


def _append_zero(sigmas: Tensor) -> Tensor:
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> Tensor:
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return _append_zero(sigmas)


def get_sigmas_exponential(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n).exp()
    return _append_zero(sigmas)


def simple_scheduler(model_sampling: DiscreteEpsSampling, steps: int) -> Tensor:
    sigs: list[float] = []
    ss = len(model_sampling.sigmas) / steps
    for x in range(steps):
        sigs.append(float(model_sampling.sigmas[-(1 + int(x * ss))]))
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def ddim_scheduler(model_sampling: DiscreteEpsSampling, steps: int) -> Tensor:
    sigs: list[float] = []
    x = 1
    if math.isclose(float(model_sampling.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
    else:
        sigs = [0.0]
    ss = max(len(model_sampling.sigmas) // steps, 1)
    while x < len(model_sampling.sigmas):
        sigs.append(float(model_sampling.sigmas[x]))
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)


def normal_scheduler(
    model_sampling: DiscreteEpsSampling,
    steps: int,
    sgm: bool = False,
) -> Tensor:
    start = model_sampling.timestep(model_sampling.sigma_max)
    end = model_sampling.timestep(model_sampling.sigma_min)
    append_zero = True
    if sgm:
        timesteps = torch.linspace(float(start), float(end), steps + 1)[:-1]
    else:
        if math.isclose(float(model_sampling.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(float(start), float(end), steps)
    sigs = [float(model_sampling.sigma(ts)) for ts in timesteps]
    if append_zero:
        sigs.append(0.0)
    return torch.FloatTensor(sigs)


def sgm_uniform_scheduler(model_sampling: DiscreteEpsSampling, steps: int) -> Tensor:
    return normal_scheduler(model_sampling, steps, sgm=True)


def beta_scheduler(model_sampling: DiscreteEpsSampling, steps: int, alpha: float = 0.6, beta: float = 0.6) -> Tensor:
    # Kumaraswamy(a, b) PPF is a closed-form stand-in for scipy.stats.beta.ppf(a, b).
    total_timesteps = len(model_sampling.sigmas) - 1
    u = 1.0 - numpy.linspace(0, 1, steps, endpoint=False)
    u = numpy.clip(u, 1e-7, 1.0 - 1e-7)
    ts = numpy.rint(((1.0 - (1.0 - u) ** (1.0 / beta)) ** (1.0 / alpha)) * total_timesteps)
    sigs: list[float] = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs.append(float(model_sampling.sigmas[int(t)]))
        last_t = t
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def linear_quadratic_schedule(
    model_sampling: DiscreteEpsSampling,
    steps: int,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
) -> Tensor:
    if steps == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule) * float(model_sampling.sigma_max.cpu())


def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    adj_idxs = torch.arange(n, dtype=torch.float).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas


SCHEDULER_HANDLERS: dict[str, SchedulerHandler] = {
    "simple": SchedulerHandler(simple_scheduler),
    "sgm_uniform": SchedulerHandler(sgm_uniform_scheduler),
    "karras": SchedulerHandler(get_sigmas_karras, use_ms=False),
    "exponential": SchedulerHandler(get_sigmas_exponential, use_ms=False),
    "ddim_uniform": SchedulerHandler(ddim_scheduler),
    "beta": SchedulerHandler(beta_scheduler),
    "normal": SchedulerHandler(normal_scheduler),
    "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
    "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),
}


def calculate_sigmas(
    model_sampling: DiscreteEpsSampling,
    scheduler_name: str,
    steps: int,
    *,
    sampler_name: str | None = None,
) -> Tensor:
    handler = SCHEDULER_HANDLERS.get(scheduler_name)
    if handler is None:
        raise ValueError(f"Unknown scheduler {scheduler_name!r}")
    discard_penultimate = sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS
    sigma_steps = steps + 1 if discard_penultimate else steps
    if handler.use_ms:
        sigmas = handler.handler(model_sampling, sigma_steps)
    else:
        sigmas = handler.handler(
            n=sigma_steps,
            sigma_min=float(model_sampling.sigma_min),
            sigma_max=float(model_sampling.sigma_max),
        )
    if discard_penultimate:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas
