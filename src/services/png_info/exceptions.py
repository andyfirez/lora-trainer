from src.services.common.exceptions import AppError


class InvalidImageError(AppError):
    status_code = 422

    def __init__(self) -> None:
        super().__init__("Invalid or unsupported image file")
