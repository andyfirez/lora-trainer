"""Business logic for saved job configs."""

from datetime import datetime, timezone
from typing import Optional, Sequence

from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.tables.job_config import ConfigType, JobConfig
from src.sampler.config import SamplingConfig
from src.services.configs.exceptions import JobConfigNotFoundError, JobConfigValidationError
from src.trainer.config import TrainConfig


class JobConfigService:
    def __init__(self, config_repo: JobConfigRepository) -> None:
        self._config_repo = config_repo

    async def list_configs(self, *, config_type: ConfigType | None = None) -> Sequence[JobConfig]:
        if config_type is not None:
            return await self._config_repo.get_by_type(config_type)
        return await self._config_repo.get_all()

    async def get_config(self, config_id: int) -> JobConfig:
        config = await self._config_repo.get_by_id(config_id)
        if config is None:
            raise JobConfigNotFoundError(config_id)
        return config

    async def create_config(
        self,
        *,
        name: str,
        config_type: ConfigType,
        config_yaml: str,
        description: str | None = None,
    ) -> JobConfig:
        self._validate_config_yaml(config_type, config_yaml)
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
        if config_yaml is not None:
            self._validate_config_yaml(config.config_type, config_yaml)
            config.config_yaml = config_yaml
        if description is not None:
            config.description = description
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

    def _validate_config_yaml(self, config_type: ConfigType, config_yaml: str) -> None:
        try:
            if config_type == ConfigType.TRAINING:
                config = TrainConfig.from_yaml(config_yaml)
                config.validate_gpu()
            else:
                config = SamplingConfig.from_yaml(config_yaml)
                config.validate_gpu()
        except Exception as exc:
            raise JobConfigValidationError(str(exc)) from exc
