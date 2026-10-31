"""Create on-disk training workspace before the worker starts."""

from pathlib import Path

from src.db.tables.lora import Lora
from src.services.loras.paths import (
    build_training_log_path,
    build_training_loss_log_path,
    resolve_training_work_dir_from_config,
)
from src.services.runnable.exceptions import RunnableValidationError
from src.storage.paths import StorageKind, StoragePaths
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import reset_loss_log


def _work_dir_not_empty(work_dir: Path) -> bool:
    if not work_dir.exists():
        return False
    return any(work_dir.iterdir())


def _write_work_dir_config(work_dir: Path, config: TrainConfig) -> None:
    config_path = work_dir / f"{config.lora_name}_config.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")


def prepare_training_workspace(lora: Lora, config: TrainConfig) -> None:
    """Materialize work dir, on-disk config, and DB paths before queueing."""
    work_dir = resolve_training_work_dir_from_config(config)
    existing_output: Path | None = None
    if lora.output_path:
        existing_output = Path(lora.output_path).expanduser().resolve()

    if _work_dir_not_empty(work_dir):
        if existing_output is None or existing_output != work_dir.resolve():
            raise RunnableValidationError(
                f"Output folder already exists and is not empty: {work_dir}"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    _write_work_dir_config(work_dir, config)

    log_path = build_training_log_path(lora.id)  # type: ignore[arg-type]
    lora.output_path = str(work_dir.resolve())
    lora.log_path = str(log_path.resolve())

    relative_path = StoragePaths.to_relative(StorageKind.LORA, work_dir)
    if relative_path is not None:
        lora.relative_path = relative_path

    if config.logging.use_ui_logger:
        reset_loss_log(build_training_loss_log_path(work_dir))
