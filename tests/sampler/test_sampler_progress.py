from pathlib import Path
from unittest.mock import MagicMock

from src.sampler.sdxl.service import SDXLLoRASampler
from src.trainer.config import TrainConfig


def test_effective_sample_prompts_adds_trigger_words_for_lora_sampling() -> None:
    config = TrainConfig(sample_prompts=["portrait"])
    sampler = SDXLLoRASampler(
        config,
        lora_paths=[Path("one.safetensors")],
        output_dir=Path("out"),
        trigger_words=["ohwx", "person"],
    )

    assert sampler._effective_sample_prompts() == ["ohwx, person, portrait"]


def test_effective_sample_prompts_unchanged_for_base_model_sampling() -> None:
    config = TrainConfig(sample_prompts=["portrait"])
    sampler = SDXLLoRASampler(
        config,
        lora_paths=[],
        output_dir=Path("out"),
        trigger_words=["ohwx"],
    )

    assert sampler._effective_sample_prompts() == ["portrait"]

    config = TrainConfig(sample_prompts=["a", "b"], sample_steps=30)
    sampler = SDXLLoRASampler(
        config,
        lora_paths=[],
        output_dir=Path("out"),
    )

    assert sampler._total_diffusion_steps() == 60


def test_total_diffusion_steps_counts_all_loras_prompts_and_steps() -> None:
    config = TrainConfig(sample_prompts=["a", "b"], sample_steps=30)
    sampler = SDXLLoRASampler(
        config,
        lora_paths=[Path("one.safetensors"), Path("two.safetensors")],
        output_dir=Path("out"),
    )

    assert sampler._total_diffusion_steps() == 120


def test_report_diffusion_progress_updates_global_step() -> None:
    config = TrainConfig(sample_prompts=["a", "b"], sample_steps=30)
    progress_callback = MagicMock()
    sampler = SDXLLoRASampler(
        config,
        lora_paths=[Path("one.safetensors")],
        output_dir=Path("out"),
        progress_callback=progress_callback,
    )

    sampler._report_diffusion_progress(completed_images=0, prompt_index=1, diffusion_step=15)

    progress_callback.assert_called_once_with(45, 60)
