"""Datasets API assembled from CRUD, images, tags, and preprocess sub-routers."""

from fastapi import APIRouter

from src.api.routers.datasets_crud import router as crud_router
from src.api.routers.datasets_images import router as images_router
from src.api.routers.datasets_preprocess import router as preprocess_router
from src.api.routers.datasets_tags import router as tags_router

router = APIRouter(prefix="/datasets", tags=["datasets"])
router.include_router(crud_router)
router.include_router(images_router)
router.include_router(tags_router)
router.include_router(preprocess_router)
