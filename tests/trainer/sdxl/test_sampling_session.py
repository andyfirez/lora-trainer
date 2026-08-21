from unittest.mock import MagicMock

import torch
from src.trainer.inference_config import SDXLInferenceConfig
from src.trainer.sdxl.latent_sampling.runner import (
    _ensure_unet_on_device,
    _restore_unet_after_decode,
)
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession


def _inference_config(**kwargs: object) -> SDXLInferenceConfig:
    return SDXLInferenceConfig(**kwargs)


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
        config=_inference_config(),
    )

    scheduler.set_timesteps.assert_called_once_with(30, device=torch.device("cpu"))
    assert session.sample_steps == 30
    assert session.vae_scale_factor == 8
    assert session.add_time_ids.shape == (1, 6)
    assert torch.equal(session.add_time_ids[0, :2], torch.tensor([1024.0, 1024.0]))
    unet.eval.assert_called_once()
    vae.enable_tiling.assert_called_once()


def test_sampling_session_uses_reference_add_time_ids_when_provided() -> None:
    scheduler = MagicMock()
    scheduler.timesteps = torch.tensor([999, 500, 0])
    unet = MagicMock()
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config.force_upcast = False
    reference = (918.0, 1216.0, 3.0, 0.0, 1024.0, 768.0)

    session = SDXLSamplingSession.create(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        device=torch.device("cpu"),
        width=768,
        height=1024,
        sample_steps=30,
        autocast_dtype=torch.float16,
        config=_inference_config(),
        reference_add_time_ids=reference,
    )

    assert session.add_time_ids.shape == (1, 6)
    assert torch.allclose(session.add_time_ids[0], torch.tensor(reference, dtype=torch.float32))
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
        config=_inference_config(
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
        config=_inference_config(
            sample_vae_fp32=False,
            sample_vae_tiling=True,
        ),
    )

    vae.enable_tiling.assert_called_once()
    assert session.vae_tiling_enabled is True


def _session_with_unet_on(device: torch.device) -> SDXLSamplingSession:
    unet = MagicMock()
    param = MagicMock()
    param.device = device
    unet.parameters.side_effect = lambda: iter([param])
    return SDXLSamplingSession(
        device=torch.device("cuda:0"),
        unet=unet,
        vae=MagicMock(),
        scheduler=MagicMock(),
        timesteps=torch.tensor([999]),
        add_time_ids=torch.zeros(1, 6),
        vae_scale_factor=8,
        autocast_dtype=torch.float16,
        sample_steps=1,
        sample_vae_fp32=False,
    )


def test_restore_unet_after_decode_moves_unet_back_to_session_device() -> None:
    session = _session_with_unet_on(torch.device("cpu"))

    _restore_unet_after_decode(session, MagicMock(), prefix="[sample 1/1]")

    session.unet.to.assert_called_once_with(torch.device("cuda:0"))


def test_restore_unet_after_decode_skips_when_already_on_session_device() -> None:
    session = _session_with_unet_on(torch.device("cuda:0"))
    log = MagicMock()

    _restore_unet_after_decode(session, log, prefix="[sample 1/1]")

    session.unet.to.assert_not_called()


def test_ensure_unet_on_device_skips_when_already_on_session_device() -> None:
    session = _session_with_unet_on(torch.device("cuda:0"))
    log = MagicMock()

    _ensure_unet_on_device(session, log, prefix="[sample 1/1]")

    session.unet.to.assert_not_called()
    log.info.assert_not_called()


def test_ensure_unet_on_device_moves_unet_when_off_device() -> None:
    session = _session_with_unet_on(torch.device("cpu"))
    log = MagicMock()

    _ensure_unet_on_device(session, log, prefix="[sample 1/1]")

    session.unet.to.assert_called_once_with(torch.device("cuda:0"))
    log.info.assert_called_once()
