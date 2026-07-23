"""Tagging job handler."""

import sys

from src.services.jobs.handlers.base import BaseJobHandler
from src.tagger.config import TaggingConfig


class TaggingJobHandler(BaseJobHandler):
    def build_command(self, job_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.tagger.runner", "--job-id", str(job_id)]

    def validate_config_yaml(self, config_yaml: str) -> None:
        config = TaggingConfig.from_yaml(config_yaml)
        if config.dataset_id <= 0:
            raise ValueError("dataset_id is required")
        config.resolve_model_repo()
