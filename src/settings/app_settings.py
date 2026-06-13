"""Application settings loaded from config.toml and environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource
from pydantic_settings.sources import PydanticBaseSettingsSource

from src.settings.models import DatabaseSettings, ServerSettings, TrainingSettings


def _default_config_path() -> Path:
    return Path(os.environ.get("APP_CONFIG_FILE", "config.toml"))


class AppSettings(BaseSettings):
    """Main application settings loaded from env, .env, and config.toml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        config_path = _default_config_path()
        if config_path.is_file():
            sources.append(TomlConfigSettingsSource(settings_cls, config_path))
        return tuple(sources)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


settings: AppSettings = get_settings()
