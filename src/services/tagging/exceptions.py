from src.services.common.exceptions import AppError


class TaggingAlreadyRunningError(AppError):
    status_code = 409

    def __init__(self, dataset_id: int) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Autotag is already running for dataset id={dataset_id}")
