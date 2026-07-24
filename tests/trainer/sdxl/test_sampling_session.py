from unittest.mock import MagicMock

import torch
from diffusers import DDPMScheduler
from src.trainer.config import TrainConfig
from src.trainer.sdxl.latent_sampling.comfy.plan import build_comfy_sampling_plan
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession


def _train_config(**kwargs: object) -> TrainConfig:
    return TrainConfig(**kwargs)


def _sampling_plan(steps: int = 30) -> object:
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    return build_comfy_sampling_plan(
        sampler_name="euler",
        scheduler_name="simple",
        steps=steps,
        noise_scheduler=scheduler,
        device=torch.device("cpu"),
    )


def test_sampling_session_builds_comfy_plan() -> None:
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config.force_upcast = False
    plan = _sampling_plan(steps=30)

    session = SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        sampling_plan=plan,
        device=torch.device("cpu"),
        width=1024,
        height=1024,
        sample_steps=30,
        autocast_dtype=torch.float16,
        config=_train_config(),
    )

    assert session.sample_steps == 30
    assert session.sampling_plan.sampler_name == "euler"
    assert len(session.sampling_plan.sigmas) == 31
    assert session.vae_scale_factor == 8
    assert session.add_time_ids.shape == (1, 6)
    assert torch.equal(session.add_time_ids[0, :2], torch.tensor([1024.0, 1024.0]))
    unet.eval.assert_called_once()
    vae.enable_tiling.assert_called_once()


def test_sampling_session_uses_reference_add_time_ids_when_provided() -> None:
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config.force_upcast = False
    reference = (918.0, 1216.0, 3.0, 0.0, 1024.0, 768.0)

    session = SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        sampling_plan=_sampling_plan(),
        device=torch.device("cpu"),
        width=768,
        height=1024,
        sample_steps=30,
        autocast_dtype=torch.float16,
        config=_train_config(),
        reference_add_time_ids=reference,
    )

    assert session.add_time_ids.shape == (1, 6)
    assert torch.allclose(session.add_time_ids[0], torch.tensor(reference, dtype=torch.float32))


def test_sampling_session_skips_vae_tiling_when_disabled() -> None:
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float16
    vae.config.force_upcast = True

    SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        sampling_plan=_sampling_plan(steps=10),
        device=torch.device("cpu"),
        width=512,
        height=512,
        sample_steps=10,
        autocast_dtype=torch.float16,
        config=_train_config(
            sample_vae_fp32=True,
            sample_vae_tiling=False,
        ),
    )

    vae.to.assert_called_once_with(dtype=torch.float32)
    vae.enable_tiling.assert_not_called()


def test_sampling_session_enables_vae_tiling_for_large_decode() -> None:
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float16
    vae.config.force_upcast = False

    session = SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        sampling_plan=_sampling_plan(steps=10),
        device=torch.device("cpu"),
        width=832,
        height=1216,
        sample_steps=10,
        autocast_dtype=torch.float16,
        config=_train_config(
            sample_vae_fp32=False,
            sample_vae_tiling=True,
        ),
    )

    vae.enable_tiling.assert_called_once()
    assert session.vae_tiling_enabled is True
