"""Exceptions shared by LoraService and SamplingService for lifecycle operations."""


class RunnableNotFoundError(Exception):
    def __init__(self, kind: str, entity_id: int) -> None:
        super().__init__(f"{kind} with id={entity_id} not found")
        self.kind = kind
        self.entity_id = entity_id


class RunnableAlreadyQueuedError(Exception):
    def __init__(self, kind: str, entity_id: int) -> None:
        super().__init__(f"{kind} id={entity_id} is already queued or running")
        self.kind = kind
        self.entity_id = entity_id


class RunnableNotCancellableError(Exception):
    def __init__(self, kind: str, entity_id: int, status: str) -> None:
        super().__init__(f"{kind} id={entity_id} cannot be cancelled in status={status}")
        self.kind = kind
        self.entity_id = entity_id
        self.status = status


class RunnableNotResumableError(Exception):
    def __init__(self, kind: str, entity_id: int, status: str) -> None:
        super().__init__(f"{kind} id={entity_id} cannot be resumed in status={status}")
        self.kind = kind
        self.entity_id = entity_id
        self.status = status


class RunnableOperationNotSupportedError(Exception):
    def __init__(self, kind: str, entity_id: int, operation: str) -> None:
        super().__init__(f"Operation '{operation}' is not supported for {kind} id={entity_id}")
        self.kind = kind
        self.entity_id = entity_id
        self.operation = operation


class RunnableValidationError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
