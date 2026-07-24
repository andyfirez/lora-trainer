"""Sampling service exceptions."""


class SamplingLoRAPathNotFoundError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(f"LoRA path not found: {path}")
        self.path = path


class SamplingPromptsNotConfiguredError(Exception):
    def __init__(self) -> None:
        super().__init__("At least one sample prompt must be configured before running sampling")
