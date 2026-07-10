from unittest.mock import MagicMock, patchimport torchfrom PIL import Imagefrom src.trainer.sdxl.latent_sampling.session import SDXLSamplingSessionfrom src.trainer.sdxl.latent_sampling.vae_decode import decode_sdxl_latentdef _build_session() -> SDXLSamplingSession:
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config.scaling_factor = 0.13025
    vae.decode.return_value = (torch.zeros(1, 3, 8, 8),)
    return SDXLSamplingSession(
        device=torch.device("cpu"),
        unet=MagicMock(),
        vae=vae,
        scheduler=MagicMock(),
        timesteps=torch.tensor([999]),
        add_time_ids=torch.zeros(1, 6),
        vae_scale_factor=8,
        autocast_dtype=torch.float16,
        sample_steps=1,
        sample_vae_fp32=True,
    )


def test_decode_sdxl_latent_full_returns_pil_image() -> None:
    session = _build_session()
    latent = torch.zeros(1, 4, 8, 8)

    image = decode_sdxl_latent(session, latent)

    assert isinstance(image, Image.Image)
    assert image.size == (8, 8)
    session.vae.decode.assert_called_once()
    scaled = session.vae.decode.call_args[0][0]
    assert torch.allclose(scaled, latent / 0.13025)


def test_decode_sdxl_latent_full_uses_autocast_for_non_fp32_vae() -> None:
    session = _build_session()
    session.vae.dtype = torch.float16
    session.sample_vae_fp32 = False

    with patch("src.trainer.sdxl.latent_sampling.vae_decode.torch.autocast") as mock_autocast:
        mock_autocast.return_value.__enter__ = MagicMock(return_value=None)
        mock_autocast.return_value.__exit__ = MagicMock(return_value=False)
        decode_sdxl_latent(session, torch.zeros(1, 4, 4, 4))

    mock_autocast.assert_called_once()
