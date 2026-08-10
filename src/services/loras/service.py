"""Business logic for LoRA training: create, start, resume, cancel, artifacts, loss."""

from pathlib import Path
from typing import Sequence

from src.api.schemas.job_loss import JobLossResponse
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.tables.lora import Lora
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.datasets.training_validation import validate_dataset_for_training
from src.services.loras.discovery import LoraDiscoveryService
from src.services.loras.exceptions import (
    LoraCheckpointNotFoundError,
    LoraNameConflictError,
    LoraNotFoundError,
    LoraReproduceError,
)
from src.services.loras.paths import (
    lora_artifacts_exist,
    resolve_completed_lora_paths,
)
from src.services.loras.relocation import find_relocated_lora
from src.services.runnable import queue, runtime
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableNotCancellableError,
    RunnableNotResumableError,
    RunnableOperationNotSupportedError,
    RunnableValidationError,
)
from src.services.runnable.loss_log_reader import read_loss_log
from src.services.runnable.samples import list_samples_for_output_dir
from src.settings.app_settings import settings
from src.storage.config_paths import resolve_config_base_model
from src.storage.paths import StorageKind, StoragePaths
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import build_loss_log_path, reset_loss_log
from src.trainer.sdxl.checkpoint_state import find_latest_checkpoint, load_resume_state
from src.trainer.training_log import JobTrainingLogger


class LoraService:
    def __init__(self, lora_repo: LoraRepository, dataset_repo: DatasetRepository) -> None:
        self._repo = lora_repo
        self._dataset_repo = dataset_repo

    async def list_loras(self) -> Sequence[Lora]:
        StoragePaths.ensure_root(StorageKind.LORA)
        await self._sync_discovered_loras()
        loras = await self._repo.list_all()
        return [lora for lora in loras if self._is_visible(lora)]

    @staticmethod
    def _is_visible(lora: Lora) -> bool:
        if lora.status == RunnableStatus.COMPLETED:
            return lora_artifacts_exist(lora)
        return True

    async def get_lora(self, lora_id: int) -> Lora:
        lora = await self._repo.get_by_id(lora_id)
        if lora is None or not self._is_visible(lora):
            raise LoraNotFoundError(lora_id)
        return lora

    async def _sync_discovered_loras(self) -> None:
        discovered = LoraDiscoveryService().discover_lora_work_dirs()
        if not discovered:
            return
        all_loras = list(await self._repo.list_all())
        existing_paths: set[str] = {lora.relative_path for lora in all_loras if lora.relative_path}
        stale_loras = [lora for lora in all_loras if lora.relative_path and not lora_artifacts_exist(lora)]
        changed = False

        for item in discovered:
            if item.relative_path in existing_paths:
                continue

            relocated = find_relocated_lora(stale_loras, item)
            if relocated is not None:
                relocated.relative_path = item.relative_path
                relocated.weights_relpath = item.weights_relpath
                self._repo._session.add(relocated)
                stale_loras.remove(relocated)
                existing_paths.add(item.relative_path)
                changed = True
                continue

            name = item.name
            candidate = name
            suffix = 2
            while await self._repo.get_by_name(candidate) is not None:
                candidate = f"{name}-{suffix}"
                suffix += 1
            await self._repo.add(
                Lora(
                    name=candidate,
                    relative_path=item.relative_path,
                    weights_relpath=item.weights_relpath,
                    base_model_name="unknown",
                    status=RunnableStatus.COMPLETED,
                    config_yaml=None,
                    output_path=str(StoragePaths.resolve(StorageKind.LORA, item.relative_path)),
                )
            )
            existing_paths.add(item.relative_path)

        if changed:
            await self._repo._session.flush()

    async def _validate_config(self, config: TrainConfig) -> None:
        try:
            if not config.base_model_name:
                raise RunnableValidationError("base_model_name is required")
            resolve_config_base_model(config.base_model_name)
            StoragePaths.resolve_lora_path(config.output_dir or "")
            if not config.concepts:
                raise RunnableValidationError("At least one training concept is required")
            for concept in config.concepts:
                dataset = await self._dataset_repo.get_by_id(concept.dataset_id)
                if dataset is None:
                    raise RunnableValidationError(f"Dataset with id={concept.dataset_id} not found")
                validate_dataset_for_training(
                    dataset,
                    config.resolution,
                    enable_bucket=config.enable_bucket,
                )
            config.validate_gpu()
        except RunnableValidationError:
            raise
        except Exception as exc:
            raise RunnableValidationError(str(exc)) from exc

    async def _create_from_config(self, name: str, config: TrainConfig, *, resolve_gpu: bool) -> Lora:
        existing = await self._repo.get_by_name(name)
        if existing is not None:
            raise LoraNameConflictError(name)
        snapshot = config.with_resolved_gpu(settings.gpu_defaults) if resolve_gpu else config
        # lora_name drives the on-disk work directory — tie it to the (unique) entity name
        # so two Lora rows never collide on the filesystem.
        snapshot = snapshot.model_copy(update={"lora_name": name})
        await self._validate_config(snapshot)
        lora = Lora(
            name=name,
            base_model_name=snapshot.base_model_name or "unknown",
            config_yaml=snapshot.to_snapshot_yaml(),
            status=RunnableStatus.DRAFT,
        )
        return await self._repo.add(lora)

    async def create_lora(self, *, name: str, config_yaml: str) -> Lora:
        config = TrainConfig.from_yaml(config_yaml)
        return await self._create_from_config(name, config, resolve_gpu=True)

    async def reproduce(self, lora_id: int, *, name: str) -> Lora:
        source = await self.get_lora(lora_id)
        if not source.config_yaml:
            raise LoraReproduceError(lora_id)
        config = TrainConfig.from_snapshot_yaml(source.config_yaml)
        return await self._create_from_config(name, config, resolve_gpu=False)

    def _reset_training_progress(self, lora: Lora) -> None:
        lora.progress_step = None
        lora.progress_total = None
        lora.progress_loss = None
        lora.progress_avr_loss = None
        lora.progress_epoch = None
        lora.progress_epoch_total = None
        lora.cache_progress_step = None
        lora.cache_progress_total = None
        lora.save_checkpoint_requested = False

    async def enqueue_lora(self, lora_id: int) -> Lora:
        lora = await self.get_lora(lora_id)
        if lora.status in (RunnableStatus.QUEUED, RunnableStatus.RUNNING):
            raise RunnableAlreadyQueuedError("Lora", lora_id)
        if not lora.config_yaml:
            raise RunnableOperationNotSupportedError("Lora", lora_id, "start")
        config = TrainConfig.from_snapshot_yaml(lora.config_yaml)
        await self._validate_config(config)
        runtime.clear_runtime(lora)
        self._reset_training_progress(lora)
        lora.resume_checkpoint_path = None
        lora.resume_from_epoch = None
        lora.resume_from_step = None
        if config.logging.use_ui_logger:
            reset_loss_log(build_loss_log_path(config))
        await queue.enqueue(self._repo._session, lora)
        return lora

    async def resume_lora(self, lora_id: int) -> Lora:
        lora = await self.get_lora(lora_id)
        if lora.status not in (RunnableStatus.FAILED, RunnableStatus.CANCELLED, RunnableStatus.ORPHAN):
            raise RunnableNotResumableError("Lora", lora_id, lora.status)
        if not lora.config_yaml:
            raise RunnableOperationNotSupportedError("Lora", lora_id, "resume")
        config = TrainConfig.from_snapshot_yaml(lora.config_yaml)
        work_dir = Path(config.output_dir) / config.lora_name
        checkpoint = find_latest_checkpoint(work_dir, config.lora_name, config.output_format.value)
        if checkpoint is None:
            raise LoraCheckpointNotFoundError(lora_id)
        resume_state = load_resume_state(checkpoint)
        runtime.clear_runtime(lora)
        lora.resume_checkpoint_path = str(checkpoint)
        lora.resume_from_epoch = resume_state.epoch
        lora.resume_from_step = resume_state.global_step
        lora.save_checkpoint_requested = False
        await queue.enqueue(self._repo._session, lora)
        return lora

    @staticmethod
    def _can_save_checkpoint_on_cancel(lora: Lora) -> bool:
        return lora.progress_step is not None and lora.progress_step > 0

    async def cancel_lora(self, lora_id: int, *, save_checkpoint: bool = False) -> Lora:
        lora = await self.get_lora(lora_id)
        if lora.status in (RunnableStatus.COMPLETED, RunnableStatus.FAILED, RunnableStatus.CANCELLED):
            raise RunnableNotCancellableError("Lora", lora_id, lora.status)
        if lora.status == RunnableStatus.RUNNING:
            if save_checkpoint and self._can_save_checkpoint_on_cancel(lora):
                lora.save_checkpoint_requested = True
                self._repo._session.add(lora)
                await self._repo._session.flush()
                return lora
            runtime.cancel(lora)
            lora.save_checkpoint_requested = False
            self._repo._session.add(lora)
            await self._repo._session.flush()
            return lora
        runtime.cancel(lora)
        runtime.clear_runtime(lora)
        self._repo._session.add(lora)
        await self._repo._session.flush()
        return lora

    async def get_logs(self, lora_id: int, tail: int = 500) -> list[str]:
        lora = await self.get_lora(lora_id)
        if not lora.log_path:
            return []
        return JobTrainingLogger.read_tail(Path(lora.log_path), lines=tail)

    async def get_loss(
        self,
        lora_id: int,
        *,
        key: str = "loss/loss",
        limit: int = 2000,
        since_step: int | None = None,
        stride: int = 1,
    ) -> JobLossResponse:
        lora = await self.get_lora(lora_id)
        if not lora.config_yaml:
            return JobLossResponse(key=key, keys=[], points=[])
        config = TrainConfig.from_snapshot_yaml(lora.config_yaml)
        log_path = build_loss_log_path(config)
        return read_loss_log(log_path, key=key, limit=limit, since_step=since_step, stride=stride)

    def list_samples(self, lora: Lora) -> list[tuple[Path, str, dict]]:
        if not lora.output_path:
            return []
        return list_samples_for_output_dir(Path(lora.output_path))

    async def finalize_completed_training(self, lora: Lora) -> None:
        """Resolve the final weights/work-dir paths once training finishes successfully."""
        if not lora.config_yaml:
            return
        paths = resolve_completed_lora_paths(lora)
        if paths is None:
            return
        lora.relative_path = paths.relative_path
        lora.weights_relpath = paths.weights_relpath
        lora.base_model_name = paths.base_model_name
        self._repo._session.add(lora)
        await self._repo._session.flush()
