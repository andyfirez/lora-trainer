"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exception_handlers import register_exception_handlers
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
from src.db.session import run_migrations
from src.services.worker.service import RunnableWorker
from src.settings.app_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up — running database migrations")
    await run_migrations()
    worker = RunnableWorker(echo_subprocess_output=False)
    await worker.start()
    app.state.runnable_worker = worker
    yield
    await worker.stop()
    logger.info("Shutting down")


app = FastAPI(title="LoRA Trainer API", version="0.1.0", lifespan=lifespan)

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
