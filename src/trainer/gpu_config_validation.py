"""Compatibility shim — use ``src.gpu.validation`` for new code."""

from src.gpu.validation import validate_gpu_config

__all__ = ["validate_gpu_config"]
