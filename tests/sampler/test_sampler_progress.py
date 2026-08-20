from pathlib import Path
from unittest.mock import MagicMock

from src.sampler.progress_utils import diffusion_progress_step, total_diffusion_steps
from src.trainer.config import TrainConfig


def test_sample_prompts_available_on_config() -> None:
    config = TrainConfig(sample_prompts=["ohwx, portrait"])
    assert config.sample_prompts == ["ohwx, portrait"]


def test_sample_prompts_same_for_base_model_sampling() -> None:
    config = TrainConfig(sample_prompts=["portrait"])
    assert config.sample_prompts == ["portrait"]

    config = TrainConfig(sample_prompts=["a", "b"], sample_steps=30)
    assert total_diffusion_steps(lora_count=1, prompt_count=2, sample_steps=30) == 60


def test_total_diffusion_steps_counts_all_loras_prompts_and_steps() -> None:
    config = TrainConfig(sample_prompts=["a", "b"], sample_steps=30)
    assert (
        total_diffusion_steps(
            lora_count=2,
            prompt_count=len(config.sample_prompts),
            sample_steps=config.sample_steps,
        )
        == 120
    )


def test_report_diffusion_progress_updates_global_step() -> None:
    config = TrainConfig(sample_prompts=["a", "b"], sample_steps=30)
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
