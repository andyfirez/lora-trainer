import inspect
from unittest.mock import MagicMock

from src.sampler.progress_callbacks import SamplingProgressReporter
from src.sampler.progress_labels import (
    PHASE_COMPOSING_GRIDS,
    PHASE_GENERATING,
    PHASE_LOADING_MODEL,
    PHASE_SAVING,
    PHASE_STARTING,
    PHASE_VAE_DECODING,
    format_sampling_progress_title,
)
from src.sampler.progress_utils import diffusion_progress_step, total_diffusion_steps
from src.trainer.inference_config import SDXLInferenceConfig


def test_total_diffusion_steps_counts_lora_and_prompts() -> None:
    assert total_diffusion_steps(lora_count=1, prompt_count=2, sample_steps=30) == 60


def test_total_diffusion_steps_counts_all_loras_prompts_and_steps() -> None:
    config = SDXLInferenceConfig(sample_prompts=["a", "b"], sample_steps=30)
    assert (
        total_diffusion_steps(
            lora_count=2,
            prompt_count=len(config.sample_prompts),
            sample_steps=config.sample_steps,
        )
        == 120
    )


def test_report_diffusion_progress_updates_global_step() -> None:
    config = SDXLInferenceConfig(sample_prompts=["a", "b"], sample_steps=30)
    progress_callback = MagicMock()
    step = diffusion_progress_step(
        completed_images=0,
        prompt_index=1,
        diffusion_step=15,
        sample_steps=config.sample_steps,
    )
    total = total_diffusion_steps(
        lora_count=1,
        prompt_count=len(config.sample_prompts),
        sample_steps=config.sample_steps,
    )
    progress_callback(step, total)

    progress_callback.assert_called_once_with(45, 60)


def test_format_title_starting_loading_and_composing_are_phase_only() -> None:
    assert format_sampling_progress_title(PHASE_STARTING) == "Starting"
    assert format_sampling_progress_title(PHASE_LOADING_MODEL) == "Loading model"
    assert format_sampling_progress_title(PHASE_COMPOSING_GRIDS) == "Composing grids"


def test_format_title_generating_includes_cell_and_steps() -> None:
    assert (
        format_sampling_progress_title(
            PHASE_GENERATING,
            cell_index=2,
            cell_total=5,
            step=12,
            step_total=30,
        )
        == "Generating · Cell 2/5 · 12/30"
    )


def test_format_title_generating_without_steps_keeps_cell() -> None:
    assert (
        format_sampling_progress_title(PHASE_GENERATING, cell_index=2, cell_total=5)
        == "Generating · Cell 2/5"
    )


def test_format_title_post_denoise_phases_include_cell_not_steps() -> None:
    assert (
        format_sampling_progress_title(
            PHASE_VAE_DECODING,
            cell_index=2,
            cell_total=5,
            step=30,
            step_total=30,
        )
        == "VAE decoding · Cell 2/5"
    )
    assert (
        format_sampling_progress_title(PHASE_SAVING, cell_index=2, cell_total=5, step=30, step_total=30)
        == "Saving · Cell 2/5"
    )


def test_reporter_rebuilds_title_on_phase_and_step() -> None:
    statuses: list[str | None] = []
    progress: list[tuple[int, int]] = []
    reporter = SamplingProgressReporter(statuses.append, lambda step, total: progress.append((step, total)))

    reporter.set_phase(PHASE_STARTING)
    assert statuses[-1] == "Starting"

    reporter.set_progress(0, 150)
    assert progress[-1] == (0, 150)
    assert statuses[-1] == "Starting"

    reporter.set_phase(PHASE_LOADING_MODEL)
    assert statuses[-1] == "Loading model"

    reporter.set_phase(PHASE_GENERATING, cell_index=2, cell_total=5)
    assert statuses[-1] == "Generating · Cell 2/5"

    reporter.set_progress(42, 150, cell_step=12, cell_step_total=30)
    assert progress[-1] == (42, 150)
    assert statuses[-1] == "Generating · Cell 2/5 · 12/30"

    reporter.set_phase(PHASE_VAE_DECODING)
    assert statuses[-1] == "VAE decoding · Cell 2/5"

    reporter.set_phase(PHASE_SAVING)
    assert statuses[-1] == "Saving · Cell 2/5"

    reporter.set_phase(PHASE_COMPOSING_GRIDS)
    assert statuses[-1] == "Composing grids"

    reporter.clear()
    assert statuses[-1] is None


def test_runner_phase_literals_match_progress_labels() -> None:
    from src.trainer.sdxl.latent_sampling.runner import run_sdxl_sampling_pass

    source = inspect.getsource(run_sdxl_sampling_pass)
    assert f'"{PHASE_VAE_DECODING}"' in source
    assert f'"{PHASE_SAVING}"' in source
