"""Application settings router."""

from fastapi import APIRouter

from src.api.dependencies import SettingsServiceDep
from src.api.schemas.settings import SettingsPatch, SettingsResponse

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/", response_model=SettingsResponse)
async def get_settings(service: SettingsServiceDep) -> SettingsResponse:
    return service.get_settings()


@router.patch("/", response_model=SettingsResponse)
async def patch_settings(body: SettingsPatch, service: SettingsServiceDep) -> SettingsResponse:
    return service.apply_patch(body)
