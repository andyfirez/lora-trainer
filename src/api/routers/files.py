"""Native file picker router."""

from fastapi import APIRouter
from fastapi.responses import Response

from src.api.dependencies import FilesServiceDep
from src.api.schemas.files import (
    PickPathRequest,
    PickPathResponse,
    ValidateBaseModelRequest,
    ValidateBaseModelResponse,
)
from src.services.files.exceptions import PickCancelledError
from src.storage.base_model_paths import resolve_config_base_model

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/pick", response_model=PickPathResponse, responses={204: {"description": "Dialog cancelled"}})
async def pick_path(body: PickPathRequest, service: FilesServiceDep) -> PickPathResponse | Response:
    try:
        path = await service.pick_path(body.kind, title=body.title, initial_path=body.initial_path)
    except PickCancelledError:
        return Response(status_code=204)
    return PickPathResponse(path=path)


@router.post("/validate-base-model", response_model=ValidateBaseModelResponse)
async def validate_base_model(body: ValidateBaseModelRequest) -> ValidateBaseModelResponse:
    try:
        resolve_config_base_model(body.path)
    except ValueError as exc:
        return ValidateBaseModelResponse(valid=False, message=str(exc))
    return ValidateBaseModelResponse(valid=True)
