"""Exceptions for the datasets service."""

from src.services.common.exceptions import AppError, NameConflictError, NotFoundError


class DatasetNotFoundError(NotFoundError):
    def __init__(self, dataset_id: int) -> None:
        self.dataset_id = dataset_id
        super().__init__("Dataset", dataset_id, message=f"Dataset with id={dataset_id} not found")


class DatasetNameConflictError(NameConflictError):
    def __init__(self, name: str) -> None:
        super().__init__("dataset", name, message=f"A dataset named '{name}' already exists")


class DatasetDirectoryNotFoundError(AppError):
    status_code = 404  # missing on-disk path, not a validation failure

    def __init__(self, path: str) -> None:
        super().__init__(f"Image directory not found: {path}")
        self.path = path


class DatasetImageNotFoundError(NotFoundError):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__("Image", filename, message=f"Image not found: {filename}")


class InvalidDatasetFilenameError(AppError):
    status_code = 400

    def __init__(self, filename: str) -> None:
        super().__init__(f"Invalid filename: {filename}")
        self.filename = filename


class DatasetTargetResolutionNotSetError(AppError):
    status_code = 422

    def __init__(self, dataset_id: int) -> None:
        super().__init__(f"Dataset id={dataset_id} has no target_resolution set")
        self.dataset_id = dataset_id


class DatasetNotPreparedError(AppError):
    status_code = 422

    def __init__(self, dataset_id: int, name: str, reason: str) -> None:
        super().__init__(f"Dataset '{name}' (id={dataset_id}) is not ready for training: {reason}")
        self.dataset_id = dataset_id
        self.name = name
        self.reason = reason


class DatasetResolutionMismatchError(AppError):
    status_code = 422

    def __init__(
        self,
        dataset_id: int,
        name: str,
        dataset_resolution: int,
        config_resolution: int,
    ) -> None:
        super().__init__(
            f"Dataset '{name}' (id={dataset_id}) target_resolution={dataset_resolution} "
            f"!= training resolution={config_resolution}"
        )
        self.dataset_id = dataset_id
        self.name = name
        self.dataset_resolution = dataset_resolution
        self.config_resolution = config_resolution


class DatasetPreprocessError(AppError):
    status_code = 422

    def __init__(self, message: str) -> None:
        super().__init__(message)
