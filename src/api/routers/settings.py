"""Application settings router."""

from fastapi import APIRouter, HTTPException

from src.api.schemas.settings import SettingsPatch, SettingsResponse, TrainingSystemInfo
from src.settings.app_settings import settings
from src.settings.config_persist import (
    apply_storage_settings,
    apply_training_settings,
    get_config_path,
    persist_storage_settings,
    persist_training_settings,
)
from src.settings.gpu_info import get_gpu_info

APP_VERSION = "0.1.0"

router = APIRouter(prefix="/settings", tags=["settings"])


def _build_settings_response() -> SettingsResponse:
    return SettingsResponse(
        max_concurrent_jobs=settings.training.max_concurrent_jobs,
        worker_poll_interval_seconds=settings.training.worker_poll_interval_seconds,
        server=settings.server,
        database=settings.database,
        storage=settings.storage,
        training=TrainingSystemInfo(
            logs_dir=settings.training.logs_dir,
            cancel_poll_interval_seconds=settings.training.cancel_poll_interval_seconds,
        ),
        config_file=str(get_config_path().resolve()),
        app_version=APP_VERSION,
        gpu=get_gpu_info(),
    )


@router.get("/", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    return _build_settings_response()


@router.patch("/", response_model=SettingsResponse)
async def patch_settings(body: SettingsPatch) -> SettingsResponse:
    if (
        body.max_concurrent_jobs is None
        and body.worker_poll_interval_seconds is None
        and body.datasets_root is None
        and body.base_models_root is None
        and body.lora_root is None
    ):
        raise HTTPException(status_code=422, detail="At least one setting must be provided")

    if body.max_concurrent_jobs is not None or body.worker_poll_interval_seconds is not None:
        persist_training_settings(
            max_concurrent_jobs=body.max_concurrent_jobs,
            worker_poll_interval_seconds=body.worker_poll_interval_seconds,
        )
        apply_training_settings(
            max_concurrent_jobs=body.max_concurrent_jobs,
            worker_poll_interval_seconds=body.worker_poll_interval_seconds,
        )

    if (
        body.datasets_root is not None
        or body.base_models_root is not None
        or body.lora_root is not None
    ):
        persist_storage_settings(
            datasets_root=body.datasets_root,
            base_models_root=body.base_models_root,
            lora_root=body.lora_root,
        )
        apply_storage_settings(
            datasets_root=body.datasets_root,
            base_models_root=body.base_models_root,
            lora_root=body.lora_root,
        )

    return _build_settings_response()
