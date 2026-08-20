"""Shared YAML/GPU helpers for TrainConfig and SamplingConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import yaml

from src.gpu.resolution import FORBIDDEN_GLOBAL_GPU_KEYS, strip_global_gpu_keys

if TYPE_CHECKING:
    from src.gpu.resolution import ResolvedGpuConfig
    from src.settings.models import GpuDefaultsSettings


class YamlGpuConfigMixin:
    """Common from_yaml / validate_gpu / snapshot helpers for GPU-aware config models."""

    FORBIDDEN_ENTITY_GPU_KEYS: ClassVar[frozenset[str]] = FORBIDDEN_GLOBAL_GPU_KEYS

    @classmethod
    def from_yaml(cls, yaml_str: str, *, snapshot: bool = False):
        data = yaml.safe_load(yaml_str) or {}
        if not isinstance(data, dict):
            data = {}
        if not snapshot:
            data = strip_global_gpu_keys(data)
        return cls.model_validate(data)

    @classmethod
    def from_snapshot_yaml(cls, yaml_str: str):
        return cls.from_yaml(yaml_str, snapshot=True)

    def resolve_gpu(self, defaults: GpuDefaultsSettings) -> ResolvedGpuConfig:
        raise NotImplementedError

    def with_resolved_gpu(self, defaults: GpuDefaultsSettings):
        resolved = self.resolve_gpu(defaults)
        return self.model_copy(update=resolved.as_train_fields())

    def validate_gpu(self) -> None:
        from src.gpu.validation import validate_gpu_config
        from src.settings.app_settings import settings

        resolved = self.resolve_gpu(settings.gpu_defaults)
        validate_gpu_config(
            attention_mechanism=resolved.attention_mechanism,
            mixed_precision=resolved.mixed_precision,
            vae_dtype=resolved.vae_dtype,
        )

    def _entity_yaml_data(self) -> dict[str, object]:
        raise NotImplementedError

    def to_yaml(self) -> str:
        return yaml.dump(self._entity_yaml_data(), allow_unicode=True, sort_keys=False)

    @classmethod
    def default_yaml(cls) -> str:
        return cls().to_yaml()
