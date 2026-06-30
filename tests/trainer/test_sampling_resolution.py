import pytest

from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.tables.job_config import ConfigType
from src.sampler.config import SamplingConfig
from src.services.configs.service import JobConfigService
from src.trainer.config import ConceptConfig, TrainConfig
from src.trainer.sampling_resolution import resolve_sampling_config


@pytest.mark.asyncio
async def test_resolve_sampling_config_returns_none_for_missing_id(
    session,
) -> None:
    repo = JobConfigRepository(session)

    result = await resolve_sampling_config(None, repo)

    assert result is None


@pytest.mark.asyncio
async def test_resolve_sampling_config_loads_sampling_yaml(
    config_service: JobConfigService,
    session,
) -> None:
    created = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - hello\nsample_steps: 20\n",
    )
    repo = JobConfigRepository(session)

    resolved = await resolve_sampling_config(created.id, repo)

    assert resolved is not None
    assert resolved.sample_prompts == ["hello"]
    assert resolved.sample_steps == 20


@pytest.mark.asyncio
async def test_resolve_sampling_config_raises_for_missing_config(session) -> None:
    repo = JobConfigRepository(session)

    with pytest.raises(ValueError, match="Sampling config with id=999 not found"):
        await resolve_sampling_config(999, repo)


@pytest.mark.asyncio
async def test_resolve_sampling_config_raises_for_non_sampling_type(
    config_service: JobConfigService,
    minimal_training_yaml: str,
    session,
) -> None:
    training_config = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )
    repo = JobConfigRepository(session)

    with pytest.raises(ValueError, match="is not a sampling config"):
        await resolve_sampling_config(training_config.id, repo)


def test_train_config_resolve_sampling_overlays_runtime_fields() -> None:
    train_config = TrainConfig(lora_name="demo", resolution=1024)
    sampling = SamplingConfig(
        sample_prompts=["a", "b"],
        sample_negative_prompt="bad",
        sample_steps=25,
        sample_cfg_scale=8.0,
    )

    resolved = train_config.resolve_sampling(sampling)

    assert resolved.lora_name == "demo"
    assert resolved.resolution == 1024
    assert resolved.sample_prompts == ["a", "b"]
    assert resolved.sample_negative_prompt == "bad"
    assert resolved.sample_steps == 25
    assert resolved.sample_cfg_scale == 8.0


def test_train_config_resolve_sampling_applies_trigger_words() -> None:
    train_config = TrainConfig(
        concepts=[ConceptConfig(dataset_id=1, trigger_words=["ohwx", "person"])],
    )
    sampling = SamplingConfig(sample_prompts=["portrait"])

    resolved = train_config.resolve_sampling(sampling)

    assert resolved.sample_prompts == ["ohwx, person, portrait"]


def test_train_config_to_yaml_excludes_runtime_sampling_fields() -> None:
    config = TrainConfig(
        sampling_config_id=1,
        sample_prompts=["runtime only"],
        sample_steps=99,
    )

    yaml_text = config.to_yaml()

    assert "sampling_config_id: 1" in yaml_text
    assert "sample_prompts" not in yaml_text
    assert "sample_steps" not in yaml_text
