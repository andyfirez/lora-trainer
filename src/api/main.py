"""FastAPI application entry point."""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.exception_handlers import register_exception_handlers
from src.api.lifespan import create_lifespan
from src.api.routers import (
    datasets,
    files,
    loras,
    png_info,
    samplings,
    storage,
)
from src.api.routers import (
    settings as settings_router,
)
from src.settings.app_settings import settings

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LoRA Trainer API", version=__version__, lifespan=create_lifespan())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(loras.router)
app.include_router(samplings.router)
app.include_router(datasets.router)
app.include_router(files.router)
app.include_router(settings_router.router)
app.include_router(storage.router)
app.include_router(png_info.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run(
        "src.api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
