"""Exceptions shared by LoraService and SamplingService for lifecycle operations."""

from src.services.common.exceptions import AppError


class RunnableNotFoundError(AppError):
    status_code = 404

    def __init__(self, kind: str, entity_id: int) -> None:
        super().__init__(f"{kind} with id={entity_id} not found")
        self.kind = kind
        self.entity_id = entity_id


class RunnableAlreadyQueuedError(AppError):
    status_code = 409

    def __init__(self, kind: str, entity_id: int) -> None:
        super().__init__(f"{kind} id={entity_id} is already queued or running")
        self.kind = kind
        self.entity_id = entity_id


class RunnableNotCancellableError(AppError):
    status_code = 409

    def __init__(self, kind: str, entity_id: int, status: str) -> None:
        super().__init__(f"{kind} id={entity_id} cannot be cancelled in status={status}")
        self.kind = kind
        self.entity_id = entity_id
        self.status = status


class RunnableNotResumableError(AppError):
    status_code = 409

    def __init__(self, kind: str, entity_id: int, status: str) -> None:
        super().__init__(f"{kind} id={entity_id} cannot be resumed in status={status}")
        self.kind = kind
        self.entity_id = entity_id
        self.status = status


class RunnableOperationNotSupportedError(AppError):
    status_code = 400

    def __init__(self, kind: str, entity_id: int, operation: str) -> None:
        super().__init__(f"Operation '{operation}' is not supported for {kind} id={entity_id}")
        self.kind = kind
        self.entity_id = entity_id
        self.operation = operation


class RunnableValidationError(AppError):
    status_code = 422

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
