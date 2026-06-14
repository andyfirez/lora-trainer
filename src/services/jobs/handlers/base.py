"""Base job handler interface."""

from abc import ABC, abstractmethod

from src.db.tables.job import Job


class BaseJobHandler(ABC):
    @abstractmethod
    def build_command(self, job_id: int) -> list[str]:
        """Return subprocess argv for the given job."""

    @abstractmethod
    def validate_config_yaml(self, config_yaml: str) -> None:
        """Validate config YAML for this job type."""
