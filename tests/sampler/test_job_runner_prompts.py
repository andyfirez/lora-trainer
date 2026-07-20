"""Tests for sampling job prompt resolution from source train config."""

from src.sampler.config import SamplingConfig
from src.trainer.config import ConceptConfig, TrainConfig


def test_sampling_job_does_not_bake_triggers_into_sample_prompts() -> None:
    sampling_config = SamplingConfig(sample_prompts=["portrait"])
    train_config = sampling_config.to_train_config()
    source_train_config = TrainConfig(
        clip_skip=3,
        concepts=[ConceptConfig(dataset_id=1, trigger_words=["ohwx"])],
    )

    effective = train_config.model_copy(update={"clip_skip": source_train_config.clip_skip})

    assert effective.clip_skip == 3
    assert effective.sample_prompts == ["portrait"]
