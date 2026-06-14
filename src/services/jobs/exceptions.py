"""Exceptions for the jobs service."""


class JobNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"Training job with id={job_id} not found")
        self.job_id = job_id


class JobAlreadyQueuedError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"Training job id={job_id} is already in the queue")
        self.job_id = job_id


class JobNotCancellableError(Exception):
    def __init__(self, job_id: int, status: str) -> None:
        super().__init__(f"Training job id={job_id} cannot be cancelled in status={status}")
        self.job_id = job_id
        self.status = status


class JobNotResumableError(Exception):
    def __init__(self, job_id: int, status: str) -> None:
        super().__init__(f"Training job id={job_id} cannot be resumed in status={status}")
        self.job_id = job_id
        self.status = status


class JobCheckpointNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"No checkpoint found for training job id={job_id}")
        self.job_id = job_id


class JobOperationNotSupportedError(Exception):
    def __init__(self, job_id: int, operation: str) -> None:
        super().__init__(f"Operation '{operation}' is not supported for job id={job_id}")
        self.job_id = job_id
        self.operation = operation
