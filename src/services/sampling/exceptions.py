"""Sampling service exceptions."""


class SamplingLoRAPathNotFoundError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(f"LoRA path not found: {path}")
        self.path = path


class SamplingCheckpointsNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"No intermediate checkpoints found for job {job_id}")
        self.job_id = job_id


class SamplingPromptsNotConfiguredError(Exception):
    def __init__(self) -> None:
        super().__init__("At least one sample prompt must be configured before running sampling")
