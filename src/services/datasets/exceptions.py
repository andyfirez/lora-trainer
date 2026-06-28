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
