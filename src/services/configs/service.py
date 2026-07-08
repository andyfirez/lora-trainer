"""Business logic for saved job configs."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import yaml

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_config_version_repo import JobConfigVersionRepository
from src.db.tables.job_config import ConfigType, JobConfig
from src.db.tables.job_config_version import JobConfigVersion
from src.sampler.config import SamplingConfig
from src.services.configs.exceptions import (
    JobConfigNotFoundError,
    JobConfigValidationError,
    JobConfigVersionNotFoundError,
)
from src.services.configs.versioning import (
    extract_lora_name,
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


@dataclass(frozen=True)
class JobConfigVersionSummary:
    version: int
    created_at: datetime
    lora_name: str | None


class JobConfigService:
    def __init__(
        self,
        config_repo: JobConfigRepository,
        dataset_repo: DatasetRepository,
        version_repo: JobConfigVersionRepository,
    ) -> None:
        self._config_repo = config_repo
        self._dataset_repo = dataset_repo
        self._version_repo = version_repo

    async def list_configs(self, *, config_type: ConfigType | None = None) -> Sequence[JobConfig]:
        if config_type is not None:
            configs = await self._config_repo.get_by_type(config_type)
        else:
            configs = await self._config_repo.get_all()
        return [await self._ensure_training_versioning(config) for config in configs]

    async def get_config(self, config_id: int) -> JobConfig:
        config = await self._config_repo.get_by_id(config_id)
        if config is None:
            raise JobConfigNotFoundError(config_id)
        return await self._ensure_training_versioning(config)

    async def create_config(
        self,
        *,
        name: str,
        config_type: ConfigType,
        config_yaml: str,
        description: str | None = None,
    ) -> JobConfig:
        await self._validate_config_yaml(config_type, config_yaml)
        if config_type == ConfigType.TRAINING:
            stored_yaml = normalize_training_config_yaml(config_yaml)
            job_config = JobConfig(
                name=name,
                config_type=config_type,
                config_yaml=stored_yaml,
                description=description,
                active_version=1,
            )
            job_config = await self._config_repo.add(job_config)
            await self._version_repo.add(
                JobConfigVersion(
                    config_id=job_config.id,  # type: ignore[arg-type]
                    version=1,
                    config_yaml=stored_yaml,
                )
            )
            return job_config
        job_config = JobConfig(
            name=name,
            config_type=config_type,
            config_yaml=config_yaml,
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
        if source.config_type == ConfigType.TRAINING:
            return await self.create_config(
                name=name or f"{source.name} (copy)",
                config_type=source.config_type,
                config_yaml=source.config_yaml,
                description=description if description is not None else source.description,
            )
        return await self.create_config(
            name=name or f"{source.name} (copy)",
            config_type=source.config_type,
            config_yaml=source.config_yaml,
            description=description if description is not None else source.description,
        )

    async def list_versions(self, config_id: int) -> list[JobConfigVersionSummary]:
        config = await self.get_config(config_id)
        if config.config_type != ConfigType.TRAINING:
            raise JobConfigValidationError("Version history is only available for training configs")
        versions = await self._version_repo.get_by_config_id(config_id)
        return [
            JobConfigVersionSummary(
                version=entry.version,
                created_at=entry.created_at,
                lora_name=extract_lora_name(entry.config_yaml),
            )
            for entry in versions
        ]

    async def get_version(self, config_id: int, version: int) -> JobConfigVersion:
        config = await self.get_config(config_id)
        if config.config_type != ConfigType.TRAINING:
            raise JobConfigValidationError("Version history is only available for training configs")
        entry = await self._version_repo.get_by_config_and_version(config_id, version)
        if entry is None:
            raise JobConfigVersionNotFoundError(config_id, version)
        normalized_yaml = normalize_training_config_yaml(entry.config_yaml)
        if normalized_yaml != entry.config_yaml:
            entry.config_yaml = normalized_yaml
            self._version_repo._session.add(entry)
            await self._version_repo._session.flush()
            await self._version_repo._session.refresh(entry)
        return entry

    async def _ensure_training_versioning(self, config: JobConfig) -> JobConfig:
        if config.config_type != ConfigType.TRAINING or config.id is None:
            return config
        versions = await self._version_repo.get_by_config_id(config.id)
        if versions and config.active_version is not None:
            active_entry = await self._version_repo.get_by_config_and_version(
                config.id,
                config.active_version,
            )
            if active_entry is not None:
                return await self._normalize_training_config_record(config)
        if versions:
            latest = max(versions, key=lambda entry: entry.version)
            config.active_version = latest.version
            config.config_yaml = latest.config_yaml
        else:
            stored_yaml = normalize_training_config_yaml(config.config_yaml)
            await self._version_repo.add(
                JobConfigVersion(
                    config_id=config.id,
                    version=1,
                    config_yaml=stored_yaml,
                )
            )
            config.active_version = 1
            config.config_yaml = stored_yaml
        return await self._normalize_training_config_record(config)

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
        current_version = config.active_version or 1
        new_version = current_version + 1
        stored_yaml = normalize_training_config_yaml(config_yaml)
        await self._version_repo.add(
            JobConfigVersion(
                config_id=config.id,  # type: ignore[arg-type]
                version=new_version,
                config_yaml=stored_yaml,
            )
        )
        config.active_version = new_version
        config.config_yaml = stored_yaml

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
        sampling_active = (
            config.sampling_enabled
            or config.sample_before_training
            or config.sample_every_n_epochs is not None
        )
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
