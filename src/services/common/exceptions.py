"""Shared domain exceptions.

Subclasses set ``status_code`` so the API can map them with a single handler.
Typical codes: 400 invalid input, 404 missing resource, 409 conflicting state,
422 entity exists but cannot be processed.
"""


class AppError(Exception):
    status_code: int = 500


class NotFoundError(AppError):
    status_code = 404

    def __init__(
        self,
        resource: str,
        resource_id: int | str,
        *,
        message: str | None = None,
    ) -> None:
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(message or f"{resource} {resource_id} not found")


class NameConflictError(AppError):
    status_code = 409

    def __init__(
        self,
        resource: str,
        name: str,
        *,
        message: str | None = None,
    ) -> None:
        self.resource = resource
        self.name = name
        super().__init__(message or f"A {resource} named '{name}' already exists")
