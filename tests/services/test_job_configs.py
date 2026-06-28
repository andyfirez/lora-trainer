import pytest
from unittest.mock import patch

from src.db.tables.job_config import ConfigType
from src.services.configs.exceptions import JobConfigNotFoundError, JobConfigValidationError
from src.services.configs.service import JobConfigService


@pytest.mark.asyncio
async def test_list_configs_returns_all(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )
    await config_service.create_config(
        name="sample",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - prompt\n",
    )

    configs = await config_service.list_configs()

    assert len(configs) == 2
    assert {config.name for config in configs} == {"train", "sample"}


@pytest.mark.asyncio
async def test_list_configs_filters_by_type(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )
    await config_service.create_config(
        name="sample",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - prompt\n",
    )

    training_configs = await config_service.list_configs(config_type=ConfigType.TRAINING)

    assert len(training_configs) == 1
    assert training_configs[0].name == "train"


@pytest.mark.asyncio
async def test_get_config_returns_entity(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    created = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
        description="demo",
    )

    fetched = await config_service.get_config(created.id)

    assert fetched.id == created.id
    assert fetched.name == "train"
    assert fetched.description == "demo"


@pytest.mark.asyncio
async def test_get_config_raises_when_missing(config_service: JobConfigService) -> None:
    with pytest.raises(JobConfigNotFoundError):
        await config_service.get_config(999)


@pytest.mark.asyncio
async def test_create_config_validates_yaml(config_service: JobConfigService) -> None:
    with pytest.raises(JobConfigValidationError):
        await config_service.create_config(
            name="bad",
            config_type=ConfigType.TRAINING,
            config_yaml="not_a_valid_config: [[[",
        )


@pytest.mark.asyncio
async def test_create_training_config_rejects_missing_dataset(
    config_service: JobConfigService,
) -> None:
    with pytest.raises(JobConfigValidationError, match="Dataset with id=999 not found"):
        await config_service.create_config(
            name="bad dataset",
            config_type=ConfigType.TRAINING,
            config_yaml="""base_model_name: x
concepts:
  - dataset_id: 999
""",
        )


@pytest.mark.asyncio
async def test_create_training_config_rejects_legacy_image_dir(
    config_service: JobConfigService,
) -> None:
    with pytest.raises(JobConfigValidationError, match="use dataset_id instead of image_dir"):
        await config_service.create_config(
            name="legacy concept field",
            config_type=ConfigType.TRAINING,
            config_yaml="""base_model_name: x
concepts:
  - image_dir: /tmp/images
""",
        )


@pytest.mark.asyncio
async def test_create_training_config_rejects_empty_concepts(
    config_service: JobConfigService,
) -> None:
    with pytest.raises(JobConfigValidationError, match="At least one training concept is required"):
        await config_service.create_config(
            name="no concepts",
            config_type=ConfigType.TRAINING,
            config_yaml="base_model_name: x\nconcepts: []\n",
        )


@pytest.mark.asyncio
async def test_create_training_config_with_valid_dataset(
    config_service: JobConfigService,
    minimal_training_yaml: str,
    training_dataset,
) -> None:
    created = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )

    assert created.name == "train"
    assert f"dataset_id: {training_dataset.id}" in created.config_yaml


@pytest.mark.asyncio
@patch("src.trainer.gpu_config_validation.torch.cuda.is_available", return_value=True)
@patch("src.trainer.gpu_config_validation.torch.cuda.get_device_capability", return_value=(7, 5))
@patch("src.trainer.gpu_config_validation.torch.cuda.get_device_name", return_value="RTX 2070")
@patch("src.trainer.gpu_config_validation.torch.cuda.is_bf16_supported", return_value=False)
@patch("src.trainer.gpu_config_validation.is_xformers_available", return_value=True)
async def test_create_sampling_config_rejects_unsupported_gpu_settings(
    _xformers: object,
    _bf16: object,
    _name: object,
    _capability: object,
    _cuda: object,
    config_service: JobConfigService,
) -> None:
    with pytest.raises(JobConfigValidationError, match="mixed_precision=bfloat16 is not supported"):
        await config_service.create_config(
            name="bad gpu settings",
            config_type=ConfigType.SAMPLING,
            config_yaml=(
                "sample_prompts:\n  - test\n"
                "attention_mechanism: xformers\n"
                "mixed_precision: bfloat16\n"
            ),
        )


@pytest.mark.asyncio
async def test_update_config_changes_fields(
    config_service: JobConfigService,
    minimal_training_yaml: str,
    training_dataset,
) -> None:
    created = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )

    updated_yaml = f"""base_model_name: y
concepts:
  - dataset_id: {training_dataset.id}
"""
    updated = await config_service.update_config(
        created.id,
        name="renamed",
        config_yaml=updated_yaml,
        description="updated",
    )

    assert updated.name == "renamed"
    assert "base_model_name: y" in updated.config_yaml
    assert updated.description == "updated"


@pytest.mark.asyncio
async def test_delete_config_removes_entity(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    created = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
    )

    await config_service.delete_config(created.id)

    with pytest.raises(JobConfigNotFoundError):
        await config_service.get_config(created.id)


@pytest.mark.asyncio
async def test_clone_config_creates_copy(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    created = await config_service.create_config(
        name="original",
        config_type=ConfigType.TRAINING,
        config_yaml=minimal_training_yaml,
        description="demo",
    )

    cloned = await config_service.clone_config(created.id)

    assert cloned.id != created.id
    assert cloned.name == "original (copy)"
    assert cloned.config_type == ConfigType.TRAINING
    assert cloned.config_yaml == created.config_yaml
    assert cloned.description == "demo"


@pytest.mark.asyncio
async def test_clone_config_custom_name(config_service: JobConfigService) -> None:
    created = await config_service.create_config(
        name="original",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - prompt\n",
    )

    cloned = await config_service.clone_config(created.id, name="custom name")

    assert cloned.name == "custom name"
    assert cloned.config_yaml == created.config_yaml


@pytest.mark.asyncio
async def test_clone_config_raises_when_missing(config_service: JobConfigService) -> None:
    with pytest.raises(JobConfigNotFoundError):
        await config_service.clone_config(999)


@pytest.mark.asyncio
async def test_create_training_config_rejects_inline_sample_prompts(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    with pytest.raises(JobConfigValidationError, match="Inline sampling parameters|Deprecated or inline"):
        await config_service.create_config(
            name="inline sampling",
            config_type=ConfigType.TRAINING,
            config_yaml=f"{minimal_training_yaml}sample_prompts:\n  - test\n",
        )


@pytest.mark.asyncio
async def test_create_training_config_requires_sampling_config_when_enabled(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    with pytest.raises(JobConfigValidationError, match="sampling_config_id is required"):
        await config_service.create_config(
            name="missing sampling ref",
            config_type=ConfigType.TRAINING,
            config_yaml=f"{minimal_training_yaml}sampling_enabled: true\n",
        )


@pytest.mark.asyncio
async def test_create_training_config_rejects_sampling_without_checkpointing(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - prompt\n",
    )
    with pytest.raises(JobConfigValidationError, match="Sampling requires checkpointing"):
        await config_service.create_config(
            name="sampling without checkpointing",
            config_type=ConfigType.TRAINING,
            config_yaml=(
                f"{minimal_training_yaml}"
                f"checkpointing_enabled: false\n"
                f"sampling_enabled: true\n"
                f"sampling_config_id: {sampling_config.id}\n"
            ),
        )


@pytest.mark.asyncio
async def test_create_training_config_rejects_deprecated_sample_after_training(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    with pytest.raises(JobConfigValidationError, match="sample_after_training"):
        await config_service.create_config(
            name="deprecated flag",
            config_type=ConfigType.TRAINING,
            config_yaml=f"{minimal_training_yaml}sample_after_training: true\n",
        )


@pytest.mark.asyncio
async def test_create_training_config_rejects_invalid_sampling_config_id(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    with pytest.raises(JobConfigValidationError, match="Sampling config with id=999 not found"):
        await config_service.create_config(
            name="bad sampling ref",
            config_type=ConfigType.TRAINING,
            config_yaml=f"{minimal_training_yaml}sampling_config_id: 999\n",
        )


@pytest.mark.asyncio
async def test_create_training_config_with_sampling_config_id(
    config_service: JobConfigService,
    minimal_training_yaml: str,
) -> None:
    sampling_config = await config_service.create_config(
        name="sampling",
        config_type=ConfigType.SAMPLING,
        config_yaml="sample_prompts:\n  - prompt\n",
    )
    created = await config_service.create_config(
        name="train",
        config_type=ConfigType.TRAINING,
        config_yaml=f"{minimal_training_yaml}sampling_config_id: {sampling_config.id}\n",
    )

    assert f"sampling_config_id: {sampling_config.id}" in created.config_yaml
    assert "sample_prompts" not in created.config_yaml
