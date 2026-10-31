"""Best-effort filesystem cleanup for deleted LoRA runs."""

import logging
import shutil
from pathlib import Path

from src.trainer.metric_logger import reset_loss_log

logger = logging.getLogger(__name__)


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        logger.warning("Failed to remove file %s", path, exc_info=True)


def cleanup_lora_artifacts(
    *,
    output_path: str | None,
    log_path: str | None,
    loss_log_path: Path | None,
) -> None:
    if output_path:
        work_dir = Path(output_path)
        try:
            if work_dir.is_dir():
                shutil.rmtree(work_dir)
        except OSError:
            logger.warning("Failed to remove work dir %s", work_dir, exc_info=True)

    if log_path:
        _unlink_if_exists(Path(log_path))

    if loss_log_path is not None:
        try:
            reset_loss_log(loss_log_path)
        except OSError:
            logger.warning("Failed to remove loss log %s", loss_log_path, exc_info=True)
