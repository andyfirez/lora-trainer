from unittest.mock import MagicMock

import torch
from src.trainer.sdxl.latent_sampling.ksample import ksample_sdxl_latent
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession
from src.trainer.sdxl.sampling import SamplePromptEmbeds


def _build_session(*, sample_steps: int = 3) -> SDXLSamplingSession:
    scheduler = MagicMock()
    scheduler.init_noise_sigma = 1.0
    scheduler.timesteps = torch.tensor([999, 500, 0][:sample_steps])

    def _scale_model_input(sample: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
        return sample

    def _step(
        noise_pred: torch.Tensor,
        _timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
        **_: object,
    ) -> tuple[torch.Tensor]:
        assert return_dict is False
        return (sample - noise_pred * 0.01,)

    scheduler.scale_model_input.side_effect = _scale_model_input
    scheduler.step.side_effect = _step

    unet = MagicMock()
    unet.config.in_channels = 4
    unet.return_value = (torch.zeros(2, 4, 8, 8),)

    return SDXLSamplingSession(
        device=torch.device("cpu"),
        unet=unet,
        vae=MagicMock(),
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        add_time_ids=torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32),
        vae_scale_factor=8,
        autocast_dtype=torch.float32,
        sample_steps=sample_steps,
        sample_vae_fp32=True,
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
