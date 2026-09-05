"""k-diffusion sample_* solvers (subset of ComfyUI comfy/k_diffusion/sampling.py)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy
import torch
from torch import Tensor

KDiffusionCallback = Callable[[dict[str, Any]], None]
KDiffusionModel = Callable[..., Tensor]


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims; cannot append to reach {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from: Tensor, sigma_to: Tensor, eta: float = 1.0) -> tuple[Tensor, Tensor]:
    if not eta:
        return sigma_to, sigma_from.new_zeros(())
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2).sqrt(),
    )
    sigma_down = (sigma_to**2 - sigma_up**2).sqrt()
    return sigma_down, sigma_up


def default_noise_sampler(x: Tensor, seed: int | None = None) -> Callable[[Tensor, Tensor], Tensor]:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed))
    return lambda _sigma, _sigma_next: torch.randn(
        x.size(),
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        generator=generator,
    )


def _sigma_from_t(t: Tensor) -> Tensor:
    return t.neg().exp()


def _t_from_sigma(sigma: Tensor) -> Tensor:
    return sigma.log().neg()


def _linear_multistep_coeff(order: int, t: numpy.ndarray, i: int, j: int) -> float:
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau: float) -> float:
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    a = float(t[i])
    b = float(t[i + 1])
    xs = numpy.linspace(a, b, 33)
    ys = numpy.array([fn(float(x)) for x in xs])
    trapz = getattr(numpy, "trapezoid", numpy.trapz)
    return float(trapz(ys, xs))


@torch.no_grad()
def sample_euler(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        gamma = 0.0
        sigma_hat = sigmas[i]
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        x = x + d * (sigmas[i + 1] - sigma_hat)
    return x


@torch.no_grad()
def sample_euler_ancestral(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed")
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if float(sigma_down) == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            x = x + d * (sigma_down - sigmas[i]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        gamma = 0.0
        sigma_hat = sigmas[i]
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1].item() == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            x = x + (d + d_2) / 2 * dt
    return x


@torch.no_grad()
def sample_lms(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
    order: int = 4,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds: list[Tensor] = []
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1].item() == 0:
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [_linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * deriv for coeff, deriv in zip(coeffs, reversed(ds)))
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed")
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if float(sigma_down) == 0:
            d = to_d(x, sigmas[i], denoised)
            x = x + d * (sigma_down - sigmas[i])
        else:
            t, t_next = _t_from_sigma(sigmas[i]), _t_from_sigma(sigma_down)
            h = t_next - t
            s = t + 0.5 * h
            x_2 = (_sigma_from_t(s) / _sigma_from_t(t)) * x - (-h * 0.5).expm1() * denoised
            denoised_2 = model(x_2, _sigma_from_t(s) * s_in, **extra_args)
            x = (_sigma_from_t(t_next) / _sigma_from_t(t)) * x - (-h).expm1() * denoised_2
        if sigmas[i + 1].item() > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Callable[[Tensor, Tensor], Tensor] | None = None,
    r: float = 0.5,
) -> Tensor:
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed")
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1].item() == 0:
            x = denoised
            continue
        lambda_s, lambda_t = _t_from_sigma(sigmas[i]), _t_from_sigma(sigmas[i + 1])
        h = lambda_t - lambda_s
        lambda_s_1 = lambda_s + r * h
        fac = 1 / (2 * r)
        sigma_s_1 = _sigma_from_t(lambda_s_1)
        alpha_s = sigmas[i] * lambda_s.exp()
        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()
        sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp(), eta)
        lambda_s_1_ = sd.log().neg()
        h_ = lambda_s_1_ - lambda_s
        x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
        if eta > 0 and s_noise > 0:
            x_2 = x_2 + alpha_s_1 * noise_sampler(sigmas[i], sigma_s_1) * s_noise * su
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)
        sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp(), eta)
        lambda_t_ = sd.log().neg()
        h_ = lambda_t_ - lambda_s
        denoised_d = (1 - fac) * denoised + fac * denoised_2
        x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
        if eta > 0 and s_noise > 0:
            x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(
    model: KDiffusionModel,
    x: Tensor,
    sigmas: Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: KDiffusionCallback | None = None,
    disable: bool | None = None,  # noqa: ARG001
) -> Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = _t_from_sigma(sigmas[i]), _t_from_sigma(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1].item() == 0:
            x = (_sigma_from_t(t_next) / _sigma_from_t(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - _t_from_sigma(sigmas[i - 1])
            ratio = h_last / h
            denoised_d = (1 + 1 / (2 * ratio)) * denoised - (1 / (2 * ratio)) * old_denoised
            x = (_sigma_from_t(t_next) / _sigma_from_t(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


SAMPLE_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "euler": sample_euler,
    "euler_ancestral": sample_euler_ancestral,
    "heun": sample_heun,
    "lms": sample_lms,
    "dpmpp_2s_ancestral": sample_dpmpp_2s_ancestral,
    "dpmpp_sde": sample_dpmpp_sde,
    "dpmpp_2m": sample_dpmpp_2m,
    "ddim": sample_euler,
}


def get_sample_function(sampler_id: str) -> Callable[..., Tensor]:
    func = SAMPLE_FUNCTIONS.get(sampler_id)
    if func is None:
        raise ValueError(f"Unknown sampler {sampler_id!r}")
    return func
