"""Convert sampling config/GPU failures into API-facing validation errors."""

from typing import Any

from src.sampler.config import SamplingConfig
from src.services.common.exceptions import AppError
from src.services.runnable.exceptions import RunnableValidationError


def parse_sampling_config(data: Any, *, snapshot: bool = False) -> SamplingConfig:
    try:
        return SamplingConfig.from_snapshot(data) if snapshot else SamplingConfig.from_dict(data)
    except AppError:
        raise
    except Exception as exc:
        raise RunnableValidationError(str(exc)) from exc


def validate_sampling_gpu(config: SamplingConfig) -> None:
    try:
        config.validate_gpu()
    except AppError:
        raise
    except Exception as exc:
        raise RunnableValidationError(str(exc)) from exc
