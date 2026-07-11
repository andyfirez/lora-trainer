"""Training job handler."""

import sys

from src.services.jobs.handlers.base import BaseJobHandler
from src.trainer.config import TrainConfig


class TrainingJobHandler(BaseJobHandler):
    def build_command(self, job_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.trainer.runner", "--job-id", str(job_id)]

    def validate_config_yaml(self, config_yaml: str) -> None:
        config = TrainConfig.from_yaml(config_yaml)
        config.validate_gpu()
        if not config.concepts:
            raise ValueError("At least one training concept is required")
        for concept in config.concepts:
            if concept.dataset_id <= 0:
                raise ValueError(f"Invalid dataset_id: {concept.dataset_id}")
