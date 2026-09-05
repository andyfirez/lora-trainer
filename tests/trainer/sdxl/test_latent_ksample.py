from unittest.mock import MagicMock

import torch
from src.sampler.registry.profiles import SamplingProfile
from src.trainer.sdxl.latent_sampling.ksample import ksample_sdxl_latent
from src.trainer.sdxl.latent_sampling.model_sampling import DiscreteEpsSampling
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession
from src.trainer.sdxl.sampling import SamplePromptEmbeds


def _model_sampling() -> DiscreteEpsSampling:
    betas = torch.linspace(0.00085, 0.012, 1000)
    return DiscreteEpsSampling.from_alphas_cumprod(torch.cumprod(1.0 - betas, dim=0))


def _build_session(
    *,
    sample_steps: int = 3,
    generation_profile: SamplingProfile = SamplingProfile.COMFY,
) -> SDXLSamplingSession:
    sigmas = torch.tensor([14.0, 7.0, 1.0, 0.0][: sample_steps + 1], dtype=torch.float32)
    unet = MagicMock()
    unet.config.in_channels = 4
    unet.return_value = (torch.zeros(2, 4, 8, 8),)
    return SDXLSamplingSession(
        device=torch.device("cpu"),
        unet=unet,
        vae=MagicMock(),
        model_sampling=_model_sampling(),
        sigmas=sigmas,
        sampler_name="euler",
        scheduler_name="normal",
        add_time_ids=torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32),
        vae_scale_factor=8,
        autocast_dtype=torch.float32,
        sample_steps=sample_steps,
        sample_vae_fp32=True,
        generation_profile=generation_profile,
    )


def _build_embeds() -> SamplePromptEmbeds:
    return SamplePromptEmbeds(
        prompt_embeds=torch.ones(1, 2, 4),
        pooled_prompt_embeds=torch.ones(1, 3),
        negative_prompt_embeds=torch.zeros(1, 2, 4),
        negative_pooled_prompt_embeds=torch.zeros(1, 3),
    )


def test_ksample_runs_unet_without_vae() -> None:
    session = _build_session()
    vae = session.vae

    latent = ksample_sdxl_latent(
        session,
        _build_embeds(),
        width=64,
        height=64,
        guidance_scale=7.5,
        seed=42,
        prompt_index=0,
    )

    assert latent.shape == (1, 4, 8, 8)
    assert session.unet.call_count == 3
    vae.decode.assert_not_called()


def test_ksample_calls_progress_callback_for_each_step() -> None:
    session = _build_session(sample_steps=3)
    progress: list[tuple[int, int]] = []

    ksample_sdxl_latent(
        session,
        _build_embeds(),
        width=64,
        height=64,
        guidance_scale=7.5,
        seed=42,
        prompt_index=0,
        on_step_end=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_ksample_calls_latent_callback_for_each_step() -> None:
    session = _build_session(sample_steps=2)
    latents: list[tuple[int, int, tuple[int, ...]]] = []

    ksample_sdxl_latent(
        session,
        _build_embeds(),
        width=64,
        height=64,
        guidance_scale=7.5,
        seed=1,
        prompt_index=0,
        on_latent=lambda completed, total, latent: latents.append((completed, total, tuple(latent.shape))),
    )

    assert latents == [(1, 2, (1, 4, 8, 8)), (2, 2, (1, 4, 8, 8))]


def test_ksample_builds_cfg_batch_inputs() -> None:
    session = _build_session(sample_steps=1)
    embeds = _build_embeds()

    ksample_sdxl_latent(
        session,
        embeds,
        width=64,
        height=64,
        guidance_scale=7.5,
        seed=7,
        prompt_index=2,
    )

    call_kwargs = session.unet.call_args.kwargs
    assert call_kwargs["encoder_hidden_states"].shape[0] == 2
    assert torch.equal(
        call_kwargs["encoder_hidden_states"][0],
        embeds.negative_prompt_embeds[0],
    )
    assert torch.equal(
        call_kwargs["encoder_hidden_states"][1],
        embeds.prompt_embeds[0],
    )
    assert call_kwargs["added_cond_kwargs"]["time_ids"].shape[0] == 2


def test_ksample_passes_max_denoise_matching_generation_profile() -> None:
    """Comfy profile scales initial noise by sqrt(1+sigma^2); reForge scales by sigma."""
    observed_max_denoise: list[bool] = []

    for profile, expected_max_denoise in (
        (SamplingProfile.COMFY, True),
        (SamplingProfile.REFORGE, False),
    ):
        session = _build_session(sample_steps=1, generation_profile=profile)
        original_noise_scaling = session.model_sampling.noise_scaling

        def spy_noise_scaling(sigma, noise, *, max_denoise=True):
            observed_max_denoise.append(max_denoise)
            return original_noise_scaling(sigma, noise, max_denoise=max_denoise)

        session.model_sampling.noise_scaling = spy_noise_scaling  # type: ignore[method-assign]

        ksample_sdxl_latent(
            session,
            _build_embeds(),
            width=64,
            height=64,
            guidance_scale=7.5,
            seed=1,
            prompt_index=0,
        )

        assert observed_max_denoise[-1] == expected_max_denoise


def test_ksample_creates_noise_using_the_session_generation_profile(monkeypatch) -> None:
    from src.trainer.sdxl.latent_sampling import ksample as ksample_module

    captured_profiles: list[SamplingProfile] = []
    original_create_initial_noise = ksample_module.create_initial_noise

    def spy_create_initial_noise(profile, seed, shape, device, dtype):
        captured_profiles.append(profile)
        return original_create_initial_noise(profile, seed, shape, device, dtype)

    monkeypatch.setattr(ksample_module, "create_initial_noise", spy_create_initial_noise)

    session = _build_session(sample_steps=1, generation_profile=SamplingProfile.REFORGE)
    ksample_sdxl_latent(
        session,
        _build_embeds(),
        width=64,
        height=64,
        guidance_scale=7.5,
        seed=3,
        prompt_index=0,
    )

    assert captured_profiles == [SamplingProfile.REFORGE]
