"""Sampling job handler."""

import sys

from src.services.jobs.handlers.base import BaseJobHandler
from src.sampler.config import SamplingConfig


class SamplingJobHandler(BaseJobHandler):
    def build_command(self, job_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.sampler.runner", "--job-id", str(job_id)]

    def validate_config_yaml(self, config_yaml: str) -> None:
        config = SamplingConfig.from_yaml(config_yaml)
        config.validate_gpu()
