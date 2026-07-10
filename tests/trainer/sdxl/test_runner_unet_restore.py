from unittest.mock import MagicMock

import torch
from src.trainer.sdxl.latent_sampling.runner import (
    _ensure_unet_on_device,
    _restore_unet_after_decode,
)
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession


def _session_with_unet_on(device: torch.device) -> SDXLSamplingSession:
    unet = MagicMock()
    unet.parameters.side_effect = lambda: iter([torch.zeros(1, device=device)])
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


def test_ensure_unet_on_device_skips_when_already_on_session_device() -> None:
    session = _session_with_unet_on(torch.device("cuda:0"))
    log = MagicMock()

    _ensure_unet_on_device(session, log, prefix="[sample 1/1]")

    session.unet.to.assert_not_called()
    log.info.assert_not_called()
