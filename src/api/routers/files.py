"""Native file picker router."""

from fastapi import APIRouter
from fastapi.responses import Response

from src.api.dependencies import FilesServiceDep
from src.api.schemas.files import PickPathRequest, PickPathResponse
from src.services.files.exceptions import PickCancelledError

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/pick", response_model=PickPathResponse, responses={204: {"description": "Dialog cancelled"}})
async def pick_path(body: PickPathRequest, service: FilesServiceDep) -> PickPathResponse | Response:
    try:
        path = await service.pick_path(body.kind, title=body.title, initial_path=body.initial_path)
    except PickCancelledError:
        return Response(status_code=204)
    return PickPathResponse(path=path)
