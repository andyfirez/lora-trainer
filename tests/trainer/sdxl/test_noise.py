"""Tests for profile-aware initial noise generation.

ComfyUI (`comfy/sample.py::prepare_noise`) always samples on CPU in float32
before moving to the target device. reForge's default `randn_source="GPU"`
(`modules/rng.py::create_generator`) samples directly on the target device in
the target dtype. These tests assert lora-trainer's `create_initial_noise`
reproduces that split.
"""

from __future__ import annotations

import torch
from src.sampler.registry.profiles import SamplingProfile
from src.trainer.sdxl.latent_sampling.noise import create_initial_noise

_SHAPE = (1, 4, 8, 8)


def test_create_initial_noise_shape_and_dtype_match_request() -> None:
    device = torch.device("cpu")
    for profile in (SamplingProfile.COMFY, SamplingProfile.REFORGE):
        noise = create_initial_noise(profile, 42, _SHAPE, device, torch.float32)
        assert noise.shape == _SHAPE
        assert noise.device == device
        assert noise.dtype == torch.float32


def test_create_initial_noise_is_deterministic_for_fixed_seed() -> None:
    device = torch.device("cpu")
    for profile in (SamplingProfile.COMFY, SamplingProfile.REFORGE):
        first = create_initial_noise(profile, 123, _SHAPE, device, torch.float32)
        second = create_initial_noise(profile, 123, _SHAPE, device, torch.float32)
        assert torch.equal(first, second)


def test_create_initial_noise_differs_for_different_seeds() -> None:
    device = torch.device("cpu")
    for profile in (SamplingProfile.COMFY, SamplingProfile.REFORGE):
        first = create_initial_noise(profile, 1, _SHAPE, device, torch.float32)
        second = create_initial_noise(profile, 2, _SHAPE, device, torch.float32)
        assert not torch.equal(first, second)


def test_create_initial_noise_comfy_samples_float32_on_cpu_regardless_of_target(monkeypatch) -> None:
    """Matches real ComfyUI: CPU generator + float32 sampling, then cast/move."""
    seen: dict[str, object] = {}
    real_randn = torch.randn

    def fake_randn(*args: object, **kwargs: object) -> torch.Tensor:
        seen["device"] = kwargs.get("device")
        seen["dtype"] = kwargs.get("dtype")
        return real_randn(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "randn", fake_randn)
    noise = create_initial_noise(
        SamplingProfile.COMFY,
        7,
        _SHAPE,
        torch.device("cpu"),
        torch.float16,
    )

    assert seen["device"] == "cpu"
    assert seen["dtype"] == torch.float32
    assert noise.dtype == torch.float16


def test_create_initial_noise_reforge_samples_directly_on_target_device_and_dtype(monkeypatch) -> None:
    """Matches real reForge default `randn_source="GPU"`: generator + sampling on target device."""
    seen: dict[str, object] = {}
    real_randn = torch.randn

    def fake_randn(*args: object, **kwargs: object) -> torch.Tensor:
        seen["device"] = kwargs.get("device")
        seen["dtype"] = kwargs.get("dtype")
        return real_randn(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "randn", fake_randn)
    device = torch.device("cpu")
    create_initial_noise(SamplingProfile.REFORGE, 7, _SHAPE, device, torch.float16)

    assert seen["device"] == device
    assert seen["dtype"] == torch.float16
