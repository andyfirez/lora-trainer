"""Tests for sampling job prompt resolution from source train config."""

from src.sampler.config import SamplingConfig
from src.trainer.config import ConceptConfig, TrainConfig
from src.trainer.sdxl.caption import apply_trigger_words_to_sample_prompts, collect_trigger_words


def test_sampling_job_applies_triggers_without_lora_paths() -> None:
    sampling_config = SamplingConfig(sample_prompts=["portrait"])
    train_config = sampling_config.to_train_config()
    source_train_config = TrainConfig(
        clip_skip=3,
        concepts=[ConceptConfig(dataset_id=1, trigger_words=["ohwx"])],
    )

    effective = train_config.model_copy(
        update={
            "clip_skip": source_train_config.clip_skip,
            "sample_prompts": apply_trigger_words_to_sample_prompts(
                train_config.sample_prompts,
                collect_trigger_words(source_train_config.concepts),
            ),
        }
    )

    assert effective.clip_skip == 3
    assert effective.sample_prompts == ["ohwx, portrait"]
