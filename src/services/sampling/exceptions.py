"""Sampling service exceptions."""

from src.db.tables.job import JobStatus


class SamplingRunNotFoundError(Exception):
    def __init__(self, sampling_run_id: int) -> None:
        super().__init__(f"Sampling run {sampling_run_id} not found")
        self.sampling_run_id = sampling_run_id


class SamplingRunAlreadyQueuedError(Exception):
    def __init__(self, sampling_run_id: int) -> None:
        super().__init__(f"Sampling run {sampling_run_id} is already queued")
        self.sampling_run_id = sampling_run_id


class SamplingRunNotCancellableError(Exception):
    def __init__(self, sampling_run_id: int, status: JobStatus) -> None:
        super().__init__(f"Sampling run {sampling_run_id} with status {status} cannot be cancelled")
        self.sampling_run_id = sampling_run_id
        self.status = status


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
