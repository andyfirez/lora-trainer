"""Business logic for jobs."""

from pathlib import Path
from typing import Sequence

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
from src.services.configs.exceptions import (
    JobConfigNotFoundError,
    JobConfigValidationError,
)
from src.services.configs.service import JobConfigService
from src.services.datasets.training_validation import validate_dataset_for_training
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
from src.services.jobs.samples import list_samples_for_output_dir
from src.services.jobs.sampling_jobs import (
    find_intermediate_checkpoints,
    prepare_sampling_config_lora_paths,
    resolve_sampling_lora_paths,
    resolve_sampling_output_dir,
    validate_lora_paths,
    validate_sample_prompts,
)
from src.services.loras.paths import assign_unique_training_job_yaml
from src.services.sampling.exceptions import SamplingCheckpointsNotFoundError
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
        self._dataset_repo = dataset_repo
        self._config_service = JobConfigService(
            config_repo,
            dataset_repo,
        )

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
            train_config = TrainConfig.from_yaml(config.config_yaml)
            for concept in train_config.concepts:
                dataset = await self._dataset_repo.get_by_id(concept.dataset_id)
                if dataset is None:
                    raise JobConfigValidationError(
                        f"Dataset with id={concept.dataset_id} not found"
                    )
                try:
                    validate_dataset_for_training(
                        dataset,
                        train_config.resolution,
                        enable_bucket=train_config.enable_bucket,
                    )
                except Exception as exc:
                    raise JobConfigValidationError(str(exc)) from exc
            job = Job(
                job_type=JobType.TRAINING,
                name=job_name,
                config_id=config.id,
                config_yaml=config.config_yaml,
            )
            job = await self._job_repo.add(job)
            if job.id is not None:
                job.config_yaml = assign_unique_training_job_yaml(job.config_yaml, job.id)
                self._job_repo._session.add(job)
                await self._job_repo._session.flush()
                await self._job_repo._session.refresh(job)
            return job
        sampling_config = SamplingConfig.from_yaml(config.config_yaml)
        job_lora_paths = (
            lora_paths
            if lora_paths is not None
            else await resolve_sampling_lora_paths(
                self._job_repo,
                source_job_id,
                runtime_train_config=self._runtime_train_config,
            )
        )
        sampling_config, paths = prepare_sampling_config_lora_paths(
            sampling_config,
            job_lora_paths or None,
        )
        if paths:
            validate_lora_paths(paths)
        validate_sample_prompts(sampling_config)
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
            str(
                await resolve_sampling_output_dir(
                    self._job_repo,
                    sampling_config,
                    job.id,
                    source_job_id,
                    runtime_train_config=self._runtime_train_config,
                )
            ),
        )
        return job

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
        runtime_config = self._runtime_train_config(training_job)
        if not find_intermediate_checkpoints(runtime_config):
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
            config = self._runtime_train_config(job)
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
        config = self._runtime_train_config(job)
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

    @staticmethod
    def _can_save_checkpoint_on_cancel(job: Job) -> bool:
        return (
            job.job_type == JobType.TRAINING
            and job.progress_step is not None
            and job.progress_step > 0
        )

    async def cancel_job(self, job_id: int, *, save_checkpoint: bool = False) -> Job:
        job = await self.get_job(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise JobNotCancellableError(job_id, job.status)
        if job.status == JobStatus.RUNNING:
            if save_checkpoint and self._can_save_checkpoint_on_cancel(job):
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
        config = self._runtime_train_config(job)
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

    def list_samples(self, job: Job) -> list[tuple[Path, str, dict]]:
        """Return (path, kind, metadata) tuples for sample files."""
        if not job.output_path:
            return []
        return list_samples_for_output_dir(Path(job.output_path))

    def get_sweep_manifest(self, job: Job):
        if not job.output_path:
            return None
        from src.sampler.sweep.manifest import read_manifest

        return read_manifest(Path(job.output_path))

    def sample_file_path(self, job: Job, relative_path: str) -> Path:
        if not job.output_path:
            raise JobOperationNotSupportedError(job.id, "sample file")
        base = Path(job.output_path).resolve()
        target = (base / relative_path).resolve()
        if not str(target).startswith(str(base)):
            raise JobOperationNotSupportedError(job.id, "sample file")
        if not target.is_file():
            raise JobOperationNotSupportedError(job.id, "sample file")
        return target

    def can_resume(self, job: Job) -> bool:
        return (
            job.job_type == JobType.TRAINING
            and job.status in (JobStatus.FAILED, JobStatus.CANCELLED)
            and bool(job.last_checkpoint_path)
        )

    def _runtime_train_config(self, job: Job) -> TrainConfig:
        return TrainConfig.from_yaml(job.config_yaml)
