from unittest.mock import MagicMock

import torch

from src.trainer.config import TrainConfig
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession


def _train_config(**kwargs: object) -> TrainConfig:
    return TrainConfig(**kwargs)


def test_sampling_session_sets_timesteps_once() -> None:
    scheduler = MagicMock()
    scheduler.timesteps = torch.tensor([999, 500, 0])
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config.force_upcast = False

    session = SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        device=torch.device("cpu"),
        width=1024,
        height=1024,
        sample_steps=30,
        autocast_dtype=torch.float16,
        config=_train_config(),
    )

    scheduler.set_timesteps.assert_called_once_with(30, device=torch.device("cpu"))
    assert session.sample_steps == 30
    assert session.vae_scale_factor == 8
    assert session.add_time_ids.shape == (1, 6)
    assert torch.equal(session.add_time_ids[0, :2], torch.tensor([1024.0, 1024.0]))
    unet.eval.assert_called_once()
    vae.enable_tiling.assert_called_once()


def test_sampling_session_prepares_vae_for_decode_when_fp32() -> None:
    scheduler = MagicMock()
    scheduler.timesteps = torch.tensor([999])
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float16
    vae.config.force_upcast = True

    SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
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
    scheduler = MagicMock()
    scheduler.timesteps = torch.tensor([999])
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float16
    vae.config.force_upcast = False

    session = SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
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
