"""Compatibility shim — use ``src.gpu.runtime`` for new code."""

from src.gpu.runtime import CudaRuntime, setup_cuda_runtime

__all__ = ["CudaRuntime", "setup_cuda_runtime"]
