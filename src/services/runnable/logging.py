"""Shared file logging helpers for Runnable (Lora/Sampling) subprocess runners."""

import logging
import sys
from pathlib import Path

from src.settings.app_settings import settings


def build_runnable_log_path(entity_id: int, *, prefix: str) -> Path:
    logs_dir = Path(settings.training.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{prefix}_{entity_id}.log"


def build_runnable_logger(entity_id: int, log_path: Path, *, name_prefix: str) -> logging.Logger:
    run_logger = logging.getLogger(f"{name_prefix}-{entity_id}")
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
