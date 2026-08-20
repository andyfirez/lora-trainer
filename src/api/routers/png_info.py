"""PNG Info inspection router."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas.png_info import PngInfoResponse
from src.services.png_info.service import inspect_image_bytes

router = APIRouter(tags=["png-info"])


@router.post("/png-info", response_model=PngInfoResponse)
async def inspect_png_info(file: UploadFile = File(...)) -> PngInfoResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    result = inspect_image_bytes(data)

    return PngInfoResponse(
        info=result.info,
        items=result.items,
        parameters=result.parameters,
        width=result.width,
        height=result.height,
        preview_base64=result.preview_base64,
    )
