"""Business logic for jobs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import yaml

from src.api.schemas.job_loss import JobLossResponse
from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.queue_repo import QueueRepository
from src.db.tables.job import Job, JobStatus, JobType
from src.db.tables.job_config import ConfigType
from src.db.tables.queue_entry import QueueEntry
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.services.configs.exceptions import JobConfigNotFoundError
from src.services.configs.service import JobConfigService
from src.services.jobs.exceptions import (
    JobAlreadyQueuedError,
    JobCheckpointNotFoundError,
    JobNotCancellableError,
    JobNotFoundError,
    JobNotResumableError,
    JobOperationNotSupportedError,
)
from src.services.jobs.handlers import get_job_handler
from src.services.jobs.loss_log_reader import read_loss_log
from src.services.sampling.exceptions import (
    SamplingCheckpointsNotFoundError,
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)
from src.tagger.config import TaggingConfig, TaggingMode
from src.trainer.config import TrainConfig
from src.trainer.metric_logger import build_loss_log_path, reset_loss_log
from src.trainer.sdxl.checkpoint_state import find_latest_checkpoint, load_resume_state
from src.trainer.training_log import JobTrainingLogger


class JobsService:
    def __init__(
        self,
        job_repo: JobRepository,
        queue_repo: QueueRepository,
        config_repo: JobConfigRepository,
        dataset_repo: DatasetRepository,
    ) -> None:
        self._job_repo = job_repo
        self._queue_repo = queue_repo
        self._config_service = JobConfigService(config_repo, dataset_repo)

    async def list_jobs(self, *, job_type: JobType | None = None) -> Sequence[Job]:
        if job_type is not None:
            return await self._job_repo.get_by_type(job_type)
        return await self._job_repo.get_all()

    async def list_jobs_by_source(self, source_job_id: int) -> Sequence[Job]:
        return await self._job_repo.get_by_source_job_id(source_job_id)

    async def get_job(self, job_id: int) -> Job:
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    async def create_from_config(
        self,
        config_id: int,
        *,
        name: str | None = None,
        lora_paths: list[str] | None = None,
        source_job_id: int | None = None,
    ) -> Job:
        config = await self._config_service.get_config(config_id)
        job_name = name or config.name
        if config.config_type == ConfigType.TRAINING:
            job = Job(
                job_type=JobType.TRAINING,
                name=job_name,
                config_id=config.id,
                config_yaml=config.config_yaml,
            )
        else:
            sampling_config = SamplingConfig.from_yaml(config.config_yaml)
            paths = (
                lora_paths
                if lora_paths is not None
                else await self._resolve_sampling_lora_paths(source_job_id)
            )
            if paths:
                self._validate_lora_paths(paths)
            self._validate_sample_prompts(sampling_config)
            job = Job(
                job_type=JobType.SAMPLING,
                name=job_name,
                config_id=config.id,
                config_yaml=config.config_yaml,
                lora_paths_yaml=yaml.safe_dump(paths, allow_unicode=True, sort_keys=False),
                source_job_id=source_job_id,
            )
            job = await self._job_repo.add(job)
            sampling_config = SamplingConfig.from_yaml(job.config_yaml)
            await self._job_repo.update_output_path(
                job,
                str(await self._resolve_sampling_output_dir(sampling_config, job.id, source_job_id)),
            )
            return job
        return await self._job_repo.add(job)

    async def create_tagging_job(
        self,
        *,
        dataset_id: int,
        dataset_name: str,
        image_dir: str,
        mode: str = "if_empty",
        threshold: float = 0.35,
        model: str = "wd-v1-4-convnextv2-tagger-v2",
        caption_extension: str = ".txt",
        strip_rating: bool = True,
        filenames: list[str] | None = None,
    ) -> Job:
        config = TaggingConfig(
            dataset_id=dataset_id,
            image_dir=image_dir,
            mode=TaggingMode(mode),
            threshold=threshold,
            model=model,
            caption_extension=caption_extension,
            strip_rating=strip_rating,
            filenames=filenames or [],
        )
        handler = get_job_handler(JobType.TAGGING)
        config_yaml = config.to_yaml()
        handler.validate_config_yaml(config_yaml)
        job = Job(
            job_type=JobType.TAGGING,
            name=f"{dataset_name} auto-tag",
            config_yaml=config_yaml,
        )
        return await self._job_repo.add(job)

    async def create_auto_sampling_for_training_job(self, training_job: Job) -> Job | None:
        if training_job.id is None or training_job.job_type != JobType.TRAINING:
            return None
        train_config = TrainConfig.from_yaml(training_job.config_yaml)
        if not train_config.sampling_enabled:
            return None
        if not train_config.checkpointing_enabled:
            return None
        if train_config.sampling_config_id is None:
            return None
        try:
            sampling_config_entity = await self._config_service.get_config(
                train_config.sampling_config_id
            )
        except JobConfigNotFoundError:
            return None
        if sampling_config_entity.config_type != ConfigType.SAMPLING:
            return None
        if not self._find_intermediate_checkpoints(train_config):
            raise SamplingCheckpointsNotFoundError(training_job.id)
        job = await self.create_from_config(
            sampling_config_entity.id,
            name=f"{training_job.name} post-train sampling",
            source_job_id=training_job.id,
        )
        await self.enqueue_job(job.id)
        return job

    async def delete_job(self, job_id: int) -> None:
        job = await self.get_job(job_id)
        entry = await self._queue_repo.get_by_job_id(job_id)
        if entry is not None:
            await self._queue_repo.delete(entry)
        await self._job_repo.delete(job)

    async def enqueue_job(self, job_id: int) -> QueueEntry:
        job = await self.get_job(job_id)
        return await self._enqueue_job(job, reset_runtime=True)

    async def _enqueue_job(self, job: Job, *, reset_runtime: bool) -> QueueEntry:
        if job.id is None:
            raise JobNotFoundError(-1)
        existing = await self._queue_repo.get_by_job_id(job.id)
        if existing is not None:
            raise JobAlreadyQueuedError(job.id)
        max_pos = await self._queue_repo.get_max_position()
        entry = QueueEntry(job_id=job.id, position=max_pos + 1)
        if reset_runtime:
            await self._job_repo.clear_runtime_state(job)
            if job.job_type == JobType.TRAINING:
                await self._job_repo.clear_resume_state(job)
        if job.job_type == JobType.TRAINING and reset_runtime:
            config = TrainConfig.from_yaml(job.config_yaml)
            if config.logging.use_ui_logger:
                reset_loss_log(build_loss_log_path(config))
        await self._job_repo.update_status(job, JobStatus.QUEUED)
        return await self._queue_repo.add(entry)

    async def resume_job(self, job_id: int) -> QueueEntry:
        job = await self.get_job(job_id)
        if job.job_type != JobType.TRAINING:
            raise JobOperationNotSupportedError(job_id, "resume")
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise JobNotResumableError(job_id, job.status)
        config = TrainConfig.from_yaml(job.config_yaml)
        work_dir = Path(config.output_dir) / config.lora_name
        checkpoint = find_latest_checkpoint(work_dir, config.lora_name, config.output_format.value)
        if checkpoint is None:
            raise JobCheckpointNotFoundError(job_id)
        resume_state = load_resume_state(checkpoint)
        await self._job_repo.set_resume_state(
            job,
            checkpoint_path=str(checkpoint),
            epoch=resume_state.epoch,
            step=resume_state.global_step,
        )
        await self._job_repo.request_checkpoint_save(job, False)
        return await self._enqueue_job(job, reset_runtime=False)

    async def cancel_job(self, job_id: int, *, save_checkpoint: bool = False) -> Job:
        job = await self.get_job(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise JobNotCancellableError(job_id, job.status)
        if job.status == JobStatus.RUNNING:
            if save_checkpoint and job.job_type == JobType.TRAINING:
                await self._job_repo.request_checkpoint_save(job, True)
                return job
            await self._job_repo.update_status(job, JobStatus.CANCELLED)
            if job.job_type == JobType.TRAINING:
                await self._job_repo.request_checkpoint_save(job, False)
            return await self._job_repo.clear_process_state(job)
        entry = await self._queue_repo.get_by_job_id(job_id)
        if entry is not None:
            await self._queue_repo.shift_positions_down(entry.position)
            await self._queue_repo.delete(entry)
        await self._job_repo.update_status(job, JobStatus.CANCELLED)
        if job.job_type == JobType.TRAINING:
            await self._job_repo.request_checkpoint_save(job, False)
        return await self._job_repo.clear_runtime_state(job)

    async def get_job_logs(self, job_id: int, tail: int = 500) -> list[str]:
        job = await self.get_job(job_id)
        if not job.log_path:
            return []
        return JobTrainingLogger.read_tail(Path(job.log_path), lines=tail)

    async def get_job_loss(
        self,
        job_id: int,
        *,
        key: str = "loss/loss",
        limit: int = 2000,
        since_step: int | None = None,
        stride: int = 1,
    ) -> JobLossResponse:
        job = await self.get_job(job_id)
        if job.job_type != JobType.TRAINING:
            raise JobOperationNotSupportedError(job_id, "loss")
        config = TrainConfig.from_yaml(job.config_yaml)
        log_path = build_loss_log_path(config)
        return read_loss_log(
            log_path,
            key=key,
            limit=limit,
            since_step=since_step,
            stride=stride,
        )

    def get_lora_paths(self, job: Job) -> list[str]:
        data = yaml.safe_load(job.lora_paths_yaml or "[]") or []
        return [str(path) for path in data]

    def list_samples(self, job: Job) -> list[Path]:
        if not job.output_path:
            return []
        output_dir = Path(job.output_path)
        if not output_dir.exists():
            return []
        return sorted(output_dir.glob("*.png"))

    def can_resume(self, job: Job) -> bool:
        return (
            job.job_type == JobType.TRAINING
            and job.status in (JobStatus.FAILED, JobStatus.CANCELLED)
            and bool(job.last_checkpoint_path)
        )

    def _validate_sample_prompts(self, config: SamplingConfig) -> None:
        if not config.sample_prompts:
            raise SamplingPromptsNotConfiguredError()

    def _validate_lora_paths(self, lora_paths: list[str]) -> None:
        for lora_path in lora_paths:
            path = Path(lora_path)
            if not path.is_file():
                raise SamplingLoRAPathNotFoundError(lora_path)

    async def _resolve_sampling_lora_paths(self, source_job_id: int | None) -> list[str]:
        if source_job_id is None:
            return []
        source_job = await self._job_repo.get_by_id(source_job_id)
        if source_job is None or source_job.job_type != JobType.TRAINING:
            return []
        train_config = TrainConfig.from_yaml(source_job.config_yaml)
        return [str(path) for path in self._find_intermediate_checkpoints(train_config)]

    def _find_intermediate_checkpoints(self, config: TrainConfig) -> list[Path]:
        work_dir = Path(config.output_dir) / config.lora_name
        ext = config.output_format.value
        epoch_paths = list(work_dir.glob(f"{config.lora_name}_epoch*.{ext}"))
        step_paths = list(work_dir.glob(f"{config.lora_name}_step*.{ext}"))
        return sorted(epoch_paths + step_paths, key=lambda path: (path.stat().st_mtime, path.name))

    async def _resolve_sampling_output_dir(
        self,
        sampling_config: SamplingConfig,
        job_id: int,
        source_job_id: int | None,
    ) -> Path:
        source_train_config: TrainConfig | None = None
        if source_job_id is not None:
            source_job = await self._job_repo.get_by_id(source_job_id)
            if source_job is not None and source_job.job_type == JobType.TRAINING:
                source_train_config = TrainConfig.from_yaml(source_job.config_yaml)
        return resolve_sampling_output_path(sampling_config, job_id, source_train_config)
