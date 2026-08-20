"""Sampling service exceptions."""

from src.services.common.exceptions import AppError


class SamplingLoRAPathNotFoundError(AppError):
    status_code = 404  # missing on-disk LoRA file, not a validation failure

    def __init__(self, path: str) -> None:
        super().__init__(f"LoRA path not found: {path}")
        self.path = path


class SamplingPromptsNotConfiguredError(AppError):
    status_code = 422

    def __init__(self) -> None:
        super().__init__("At least one sample prompt must be configured before running sampling")
