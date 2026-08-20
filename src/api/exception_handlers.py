"""FastAPI exception handlers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.services.common.exceptions import AppError


async def app_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AppError):
        raise exc
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
