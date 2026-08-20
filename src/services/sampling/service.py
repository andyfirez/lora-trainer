"""Business logic for sampling: create, start, cancel, results."""

from pathlib import Path
from typing import Sequence

import yaml
from src.db.repositories.sampling_repo import SamplingRepository
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.sampler.config import SamplingConfig
from src.services.runnable import queue, runtime
from src.services.runnable.artifacts import list_runnable_samples, read_runnable_logs
from src.services.runnable.exceptions import (
    RunnableAlreadyQueuedError,
    RunnableNotCancellableError,
    RunnableNotFoundError,
    RunnableOperationNotSupportedError,
    RunnableValidationError,
)
from src.services.runnable.handlers.sampling import SamplingHandler
from src.services.runnable.samples import resolve_safe_sample_file
from src.services.sampling.lora_paths import (
    prepare_sampling_config_lora_paths,
    resolve_lora_paths_from_sampling_config,
    resolve_sampling_output_dir,
    validate_lora_paths,
    validate_sample_prompts,
)
from src.settings.app_settings import settings


class SamplingService:
    def __init__(self, sampling_repo: SamplingRepository) -> None:
        self._repo = sampling_repo

    async def list_samplings(self) -> Sequence[Sampling]:
        return await self._repo.list_ordered(Sampling.created_at.desc())

    async def get_sampling(self, sampling_id: int) -> Sampling:
        sampling = await self._repo.get_by_id(sampling_id)
        if sampling is None:
            raise RunnableNotFoundError("Sampling", sampling_id)
        return sampling

    @staticmethod
    def _validate_output_dir(config: SamplingConfig) -> None:
        raw = config.output_dir.strip()
        if not raw:
            raise RunnableValidationError("output_dir is required")
        if not Path(raw).expanduser().is_absolute():
            raise RunnableValidationError("output_dir must be an absolute path")

    async def create_sampling(
        self,
        *,
        name: str,
        config_yaml: str,
        lora_paths: list[str] | None = None,
    ) -> Sampling:
        config = SamplingConfig.from_yaml(config_yaml)
        self._validate_output_dir(config)
        entity_lora_paths = (
            lora_paths if lora_paths is not None else resolve_lora_paths_from_sampling_config(config)
        )
        config, paths = prepare_sampling_config_lora_paths(config, entity_lora_paths or None)
        if paths:
            validate_lora_paths(paths)
        validate_sample_prompts(config)
        snapshot = config.with_resolved_gpu(settings.gpu_defaults)
        snapshot.validate_gpu()
        sampling = Sampling(
            name=name,
            config_yaml=snapshot.to_snapshot_yaml(),
            lora_paths_yaml=yaml.safe_dump(paths, allow_unicode=True, sort_keys=False),
            status=RunnableStatus.DRAFT,
        )
        sampling = await self._repo.add(sampling)
        output_dir = resolve_sampling_output_dir(snapshot, sampling.id)  # type: ignore[arg-type]
        sampling.output_path = str(output_dir)
        self._repo._session.add(sampling)
        await self._repo._session.flush()
        return sampling

    async def enqueue_sampling(self, sampling_id: int) -> Sampling:
        sampling = await self.get_sampling(sampling_id)
        if sampling.status in (RunnableStatus.QUEUED, RunnableStatus.RUNNING):
            raise RunnableAlreadyQueuedError("Sampling", sampling_id)
        SamplingHandler().validate_config_yaml(sampling.config_yaml)
        runtime.clear_runtime(sampling)
        sampling.progress_step = None
        sampling.progress_total = None
        sampling.progress_status = None
        await queue.enqueue(self._repo._session, sampling)
        return sampling

    async def cancel_sampling(self, sampling_id: int) -> Sampling:
        sampling = await self.get_sampling(sampling_id)
        if sampling.status in (RunnableStatus.COMPLETED, RunnableStatus.FAILED, RunnableStatus.CANCELLED):
            raise RunnableNotCancellableError("Sampling", sampling_id, sampling.status)
        runtime.cancel(sampling)
        if sampling.status != RunnableStatus.RUNNING:
            runtime.clear_runtime(sampling)
        self._repo._session.add(sampling)
        await self._repo._session.flush()
        return sampling

    async def get_logs(self, sampling_id: int, tail: int = 500) -> list[str]:
        sampling = await self.get_sampling(sampling_id)
        return read_runnable_logs(sampling, tail)

    def get_lora_paths(self, sampling: Sampling) -> list[str]:
        data = yaml.safe_load(sampling.lora_paths_yaml or "[]") or []
        return [str(path) for path in data]

    def list_samples(self, sampling: Sampling) -> list[tuple[Path, str, dict]]:
        return list_runnable_samples(sampling)

    def get_sweep_manifest(self, sampling: Sampling):
        if not sampling.output_path:
            return None
        from src.sampler.sweep.manifest import read_manifest

        return read_manifest(Path(sampling.output_path))

    def sample_file_path(self, sampling: Sampling, relative_path: str) -> Path:
        if not sampling.output_path:
            raise RunnableOperationNotSupportedError("Sampling", sampling.id or 0, "sample file")
        target = resolve_safe_sample_file(Path(sampling.output_path), relative_path)
        if target is None:
            raise RunnableOperationNotSupportedError("Sampling", sampling.id or 0, "sample file")
        return target
