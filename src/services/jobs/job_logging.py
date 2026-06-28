"""Shared file logging helpers for job subprocess runners."""

import logging
import sys
from pathlib import Path

from src.settings.app_settings import settings


def build_job_log_path(job_id: int) -> Path:
    logs_dir = Path(settings.training.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"job_{job_id}.log"


def build_job_logger(job_id: int, log_path: Path, *, name_prefix: str) -> logging.Logger:
    run_logger = logging.getLogger(f"{name_prefix}-{job_id}")
    run_logger.setLevel(logging.INFO)
    run_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    run_logger.addHandler(file_handler)
    run_logger.addHandler(stream_handler)
    run_logger.propagate = False
    return run_logger
