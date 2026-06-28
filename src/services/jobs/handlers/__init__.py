"""Job type handlers."""

from src.db.tables.job import JobType
from src.services.jobs.handlers.base import BaseJobHandler
from src.services.jobs.handlers.sampling import SamplingJobHandler
from src.services.jobs.handlers.tagging import TaggingJobHandler
from src.services.jobs.handlers.training import TrainingJobHandler

_HANDLERS: dict[JobType, BaseJobHandler] = {
    JobType.TRAINING: TrainingJobHandler(),
    JobType.SAMPLING: SamplingJobHandler(),
    JobType.TAGGING: TaggingJobHandler(),
}


def get_job_handler(job_type: JobType) -> BaseJobHandler:
    return _HANDLERS[job_type]
