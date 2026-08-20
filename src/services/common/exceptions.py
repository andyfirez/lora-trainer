"""Shared domain exceptions.

Subclasses set ``status_code`` so the API can map them with a single handler.
Typical codes: 400 invalid input, 404 missing resource, 409 conflicting state,
422 entity exists but cannot be processed.
"""


class AppError(Exception):
    status_code: int = 500
