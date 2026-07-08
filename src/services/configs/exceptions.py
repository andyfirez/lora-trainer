"""Exceptions for job config operations."""


class JobConfigNotFoundError(Exception):
    def __init__(self, config_id: int) -> None:
        self.config_id = config_id
        super().__init__(f"Job config {config_id} not found")


class JobConfigValidationError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class JobConfigVersionNotFoundError(Exception):
    def __init__(self, config_id: int, version: int) -> None:
        self.config_id = config_id
        self.version = version
        super().__init__(f"Job config {config_id} version {version} not found")
