"""GPU availability and device metadata for system settings."""

from pydantic import BaseModel


class GpuInfo(BaseModel):
    cuda_available: bool
    device_name: str | None = None
    device_count: int = 0
    vram_gb: list[float] | None = None


def get_gpu_info() -> GpuInfo:
    try:
        import torch
    except ImportError:
        return GpuInfo(cuda_available=False)

    if not torch.cuda.is_available():
        return GpuInfo(cuda_available=False)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else None
    vram_gb: list[float] = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        vram_gb.append(round(props.total_memory / (1024**3), 2))

    return GpuInfo(
        cuda_available=True,
        device_name=device_name,
        device_count=device_count,
        vram_gb=vram_gb or None,
    )
