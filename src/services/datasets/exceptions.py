"""Exceptions for the datasets service."""


class DatasetNotFoundError(Exception):
    def __init__(self, dataset_id: int) -> None:
        super().__init__(f"Dataset with id={dataset_id} not found")
        self.dataset_id = dataset_id


class DatasetNameConflictError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"A dataset named '{name}' already exists")
        self.name = name


class DatasetDirectoryNotFoundError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(f"Image directory not found: {path}")
        self.path = path


class DatasetImageNotFoundError(Exception):
    def __init__(self, filename: str) -> None:
        super().__init__(f"Image not found: {filename}")
        self.filename = filename


class InvalidDatasetFilenameError(Exception):
    def __init__(self, filename: str) -> None:
        super().__init__(f"Invalid filename: {filename}")
        self.filename = filename


class DatasetTargetResolutionNotSetError(Exception):
    def __init__(self, dataset_id: int) -> None:
        super().__init__(f"Dataset id={dataset_id} has no target_resolution set")
        self.dataset_id = dataset_id


class DatasetNotPreparedError(Exception):
    def __init__(self, dataset_id: int, name: str, reason: str) -> None:
        super().__init__(f"Dataset '{name}' (id={dataset_id}) is not ready for training: {reason}")
        self.dataset_id = dataset_id
        self.name = name
        self.reason = reason


class DatasetResolutionMismatchError(Exception):
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


class DatasetPreprocessError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
