"""Exceptions for the files service."""


class PickCancelledError(Exception):
    def __init__(self, message: str = "File picker was cancelled") -> None:
        super().__init__(message)
