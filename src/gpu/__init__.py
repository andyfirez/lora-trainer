"""GPU discovery, resolution, validation, and runtime setup."""

from src.gpu.config_mixin import YamlGpuConfigMixin
from src.gpu.discovery import GpuInfo, get_gpu_info
from src.gpu.resolution import (
    FORBIDDEN_GLOBAL_GPU_KEYS,
    ResolvedGpuConfig,
    resolve_gpu_config,
    strip_global_gpu_keys,
    strip_gpu_overrides_matching_defaults,
)
from src.gpu.validation import validate_gpu_config

__all__ = [
    "FORBIDDEN_GLOBAL_GPU_KEYS",
    "GpuInfo",
    "ResolvedGpuConfig",
    "YamlGpuConfigMixin",
    "get_gpu_info",
    "resolve_gpu_config",
    "strip_global_gpu_keys",
    "strip_gpu_overrides_matching_defaults",
    "validate_gpu_config",
]
