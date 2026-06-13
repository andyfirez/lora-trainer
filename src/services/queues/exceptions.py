"""Exceptions for the queue service."""


class QueueEntryNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"No queue entry found for job id={job_id}")
        self.job_id = job_id
