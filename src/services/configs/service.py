"""Business logic for saved job configs."""

from datetime import datetime, timezone

from typing import Sequence

import yaml
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.tables.job_config import ConfigType, JobConfig
from src.sampler.config import SamplingConfig
from src.services.configs.exceptions import (
    JobConfigNotFoundError,
    JobConfigValidationError,
)
from src.services.configs.versioning import (
    normalize_training_config_yaml,
    yaml_configs_equal,
)
from src.services.datasets.training_validation import validate_dataset_for_training
from src.trainer.config import (
    FORBIDDEN_DEPRECATED_CONCEPT_KEYS,
    FORBIDDEN_DEPRECATED_TRAIN_KEYS,
    FORBIDDEN_INLINE_SAMPLING_KEYS,
    TrainConfig,
)


class JobConfigService:
    def __init__(
        self,
        config_repo: JobConfigRepository,
        dataset_repo: DatasetRepository,
    ) -> None:
        self._config_repo = config_repo
        self._dataset_repo = dataset_repo

    async def list_configs(self, *, config_type: ConfigType | None = None) -> Sequence[JobConfig]:
        if config_type is not None:
            return await self._config_repo.get_by_type(config_type)
        return await self._config_repo.get_all()

    async def get_config(self, config_id: int) -> JobConfig:
        config = await self._config_repo.get_by_id(config_id)
        if config is None:
            raise JobConfigNotFoundError(config_id)
        if config.config_type == ConfigType.TRAINING:
            return await self._normalize_training_config_record(config)
        return config

    async def create_config(
        self,
        *,
        name: str,
        config_type: ConfigType,
        config_yaml: str,
        description: str | None = None,
    ) -> JobConfig:
        await self._validate_config_yaml(config_type, config_yaml)
        stored_yaml = (
            normalize_training_config_yaml(config_yaml)
            if config_type == ConfigType.TRAINING
            else config_yaml
        )
        job_config = JobConfig(
            name=name,
            config_type=config_type,
            config_yaml=stored_yaml,
            description=description,
        )
        return await self._config_repo.add(job_config)

    async def update_config(
        self,
        config_id: int,
        *,
        name: str | None = None,
        config_yaml: str | None = None,
        description: str | None = None,
    ) -> JobConfig:
        config = await self.get_config(config_id)
        if name is not None:
            config.name = name
        if description is not None:
            config.description = description
        if config_yaml is not None:
            if config.config_type == ConfigType.TRAINING:
                await self._update_training_config_yaml(config, config_yaml)
            else:
                await self._validate_config_yaml(config.config_type, config_yaml)
                config.config_yaml = config_yaml
        config.updated_at = datetime.now(timezone.utc)
        self._config_repo._session.add(config)
        await self._config_repo._session.flush()
        await self._config_repo._session.refresh(config)
        return config

    async def delete_config(self, config_id: int) -> None:
        config = await self.get_config(config_id)
        await self._config_repo.delete(config)

    async def clone_config(
        self,
        config_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> JobConfig:
        source = await self.get_config(config_id)
        return await self.create_config(
            name=name or f"{source.name} (copy)",
            config_type=source.config_type,
            config_yaml=source.config_yaml,
            description=description if description is not None else source.description,
        )

    async def _normalize_training_config_record(self, config: JobConfig) -> JobConfig:
        normalized_yaml = normalize_training_config_yaml(config.config_yaml)
        if normalized_yaml == config.config_yaml:
            return config
        config.config_yaml = normalized_yaml
        config.updated_at = datetime.now(timezone.utc)
        self._config_repo._session.add(config)
        await self._config_repo._session.flush()
        await self._config_repo._session.refresh(config)
        return config

    async def _update_training_config_yaml(self, config: JobConfig, config_yaml: str) -> None:
        await self._validate_config_yaml(ConfigType.TRAINING, config_yaml)
        if yaml_configs_equal(config_yaml, config.config_yaml):
            return
        config.config_yaml = normalize_training_config_yaml(config_yaml)

    async def _validate_config_yaml(self, config_type: ConfigType, config_yaml: str) -> None:
        try:
            raw = yaml.safe_load(config_yaml) or {}
            if config_type == ConfigType.TRAINING:
                forbidden = (FORBIDDEN_INLINE_SAMPLING_KEYS | FORBIDDEN_DEPRECATED_TRAIN_KEYS) & raw.keys()
                if forbidden:
                    raise JobConfigValidationError(
                        "Deprecated or inline sampling parameters are not allowed; "
                        "use sampling_enabled and sampling_config_id: "
                        + ", ".join(sorted(forbidden))
                    )
                self._validate_raw_concepts(raw.get("concepts"))
                config = TrainConfig.from_yaml(config_yaml)
                config.validate_gpu()
                await self._validate_training_config(config)
            else:
                config = SamplingConfig.from_yaml(config_yaml)
                config.validate_gpu()
        except JobConfigValidationError:
            raise
        except Exception as exc:
            raise JobConfigValidationError(str(exc)) from exc

    async def _validate_training_config(self, config: TrainConfig) -> None:
        await self._validate_training_concepts(config)
        sampling_active = config.sampling_enabled
        if config.sampling_enabled and not config.checkpointing_enabled:
            raise JobConfigValidationError("Sampling requires checkpointing to be enabled")
        if sampling_active and config.sampling_config_id is None:
            raise JobConfigValidationError(
                "sampling_config_id is required when sampling is enabled"
            )
        if config.sampling_config_id is not None:
            sampling_entity = await self._config_repo.get_by_id(config.sampling_config_id)
            if sampling_entity is None:
                raise JobConfigValidationError(
                    f"Sampling config with id={config.sampling_config_id} not found"
                )
            if sampling_entity.config_type != ConfigType.SAMPLING:
                raise JobConfigValidationError(
                    f"Job config id={config.sampling_config_id} is not a sampling config"
                )
            if sampling_active:
                sampling = SamplingConfig.from_yaml(sampling_entity.config_yaml)
                if not sampling.sample_prompts:
                    raise JobConfigValidationError(
                        "Referenced sampling config has no sample_prompts"
                    )

    def _validate_raw_concepts(self, concepts: object) -> None:
        if not isinstance(concepts, list):
            return
        for index, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                continue
            deprecated = FORBIDDEN_DEPRECATED_CONCEPT_KEYS & concept.keys()
            if deprecated:
                raise JobConfigValidationError(
                    f"Concept {index + 1}: use dataset_id instead of "
                    + ", ".join(sorted(deprecated))
                )

    async def _validate_training_concepts(self, config: TrainConfig) -> None:
        if not config.concepts:
            raise JobConfigValidationError("At least one training concept is required")
        for concept in config.concepts:
            dataset = await self._dataset_repo.get_by_id(concept.dataset_id)
            if dataset is None:
                raise JobConfigValidationError(
                    f"Dataset with id={concept.dataset_id} not found"
                )
            try:
                validate_dataset_for_training(
                    dataset,
                    config.resolution,
                    enable_bucket=config.enable_bucket,
                )
            except Exception as exc:
                raise JobConfigValidationError(str(exc)) from exc
