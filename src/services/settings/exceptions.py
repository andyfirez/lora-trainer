from src.services.common.exceptions import AppError


class EmptySettingsPatchError(AppError):
    status_code = 422

    def __init__(self) -> None:
        super().__init__("At least one setting must be provided")


class InvalidGpuDefaultsError(AppError):
    status_code = 422

    def __init__(self, message: str) -> None:
        super().__init__(message)
