"""k-diffusion sampler steps adapted from ComfyUI 0.27.0 (GPL-3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

Callback = Callable[[dict[str, Any]], None]
ModelFn = Callable[..., Tensor]


def to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    return (x - denoised) / sigma


def get_ancestral_step(sigma_from: Tensor, sigma_to: Tensor, eta: float = 1.0) -> tuple[Tensor, Tensor]:
    if not eta:
        return sigma_to, sigma_to.new_zeros(())
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x: Tensor, *, seed: int | None = None) -> Callable[[Tensor, Tensor], Tensor]:
    generator: torch.Generator | None
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed + 1)
    else:
        generator = None

    def sample(_sigma: Tensor, _sigma_next: Tensor) -> Tensor:
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

    return sample


@torch.no_grad()
def sample_euler(
    model: ModelFn,
    x: Tensor,
    sigmas: Tensor,
    *,
    extra_args: dict[str, Any] | None = None,
    callback: Callback | None = None,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma_hat = sigmas[index]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        derivative = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": index, "sigma": sigmas[index], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[index + 1] - sigma_hat
        x = x + derivative * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(
    model: ModelFn,
    x: Tensor,
    sigmas: Tensor,
    *,
    extra_args: dict[str, Any] | None = None,
    callback: Callback | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed")
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for index in range(len(sigmas) - 1):
        denoised = model(x, sigmas[index] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[index], sigmas[index + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": index,
                    "sigma": sigmas[index],
                    "sigma_hat": sigmas[index],
                    "denoised": denoised,
                }
            )
        if float(sigma_down) == 0.0:
            x = denoised
        else:
            derivative = to_d(x, sigmas[index], denoised)
            dt = sigma_down - sigmas[index]
            x = x + derivative * dt + noise_sampler(sigmas[index], sigmas[index + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2m(
    model: ModelFn,
    x: Tensor,
    sigmas: Tensor,
    *,
    extra_args: dict[str, Any] | None = None,
    callback: Callback | None = None,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    def sigma_fn(t: Tensor) -> Tensor:
        return t.neg().exp()

    def t_fn(sigma: Tensor) -> Tensor:
        return sigma.log().neg()

    old_denoised: Tensor | None = None
    for index in range(len(sigmas) - 1):
        denoised = model(x, sigmas[index] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": index,
                    "sigma": sigmas[index],
                    "sigma_hat": sigmas[index],
                    "denoised": denoised,
                }
            )
        t = t_fn(sigmas[index])
        t_next = t_fn(sigmas[index + 1])
        h = t_next - t
        if old_denoised is None or float(sigmas[index + 1]) == 0.0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[index - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


SAMPLER_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "euler": sample_euler,
    "euler_ancestral": sample_euler_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
}


def run_sampler(name: str, model: ModelFn, x: Tensor, sigmas: Tensor, **kwargs: Any) -> Tensor:
    try:
        sampler = SAMPLER_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported sampler {name!r}") from exc
    return sampler(model, x, sigmas, **kwargs)
