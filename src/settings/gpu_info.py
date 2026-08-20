"""Compatibility shim — use ``src.gpu.discovery`` for new code."""

from src.gpu.discovery import GpuInfo, get_gpu_info

__all__ = ["GpuInfo", "get_gpu_info"]
