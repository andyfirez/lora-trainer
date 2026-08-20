"""Shared CUDA runtime setup for SDXL training and sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.settings.app_settings import settings
from src.trainer.gpu_config_mixin import YamlGpuConfigMixin
from src.trainer.gpu_resolution import ResolvedGpuConfig
from src.trainer.sdxl.dtypes import weight_dtype_to_torch


@dataclass(frozen=True)
class CudaRuntime:
    device: torch.device
    gpu: ResolvedGpuConfig
    weight_dtype: torch.dtype


def setup_cuda_runtime(config: YamlGpuConfigMixin) -> CudaRuntime:
    """Validate GPU config, require CUDA, and apply tf32 settings."""
    config.validate_gpu()

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available (torch {torch.__version__}). "
            "Install GPU-enabled PyTorch: run `uv sync` in the project root "
            "after configuring the pytorch-cu130 index in pyproject.toml."
        )

    device = torch.device("cuda")
    gpu = config.resolve_gpu(settings.gpu_defaults)
    weight_dtype = weight_dtype_to_torch(gpu.mixed_precision)

    if gpu.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return CudaRuntime(device=device, gpu=gpu, weight_dtype=weight_dtype)
