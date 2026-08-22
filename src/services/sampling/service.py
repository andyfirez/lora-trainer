"""Business logic for sampling: create, start, cancel, results."""

from pathlib import Path
from typing import Any, Sequence

from src.db.repositories.sampling_repo import SamplingRepository
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import (
    effective_sampling_output_dir,
    resolve_sampling_config_output_dir,
)
from src.services.runnable.artifacts import list_runnable_samples, read_runnable_logs
from src.services.runnable.exceptions import (
    RunnableNotFoundError,
    RunnableOperationNotSupportedError,
    RunnableValidationError,
)
from src.services.runnable.handlers import get_runnable_handler
from src.services.runnable.samples import resolve_safe_sample_file
from src.services.sampling.exceptions import LivePreviewNotReadyError
from src.services.sampling.lora_paths import (
    prepare_sampling_config_lora_paths,
    resolve_lora_paths_from_sampling_config,
    resolve_sampling_output_dir,
    validate_lora_paths,
    validate_sample_prompts,
)
from src.settings.app_settings import settings
from src.trainer.sdxl.latent_sampling.preview import LIVE_PREVIEW_FILENAME


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
        try:
            resolve_sampling_config_output_dir(effective_sampling_output_dir(config))
        except ValueError as exc:
            raise RunnableValidationError(str(exc)) from exc

    async def create_sampling(
        self,
        *,
        name: str,
        config: dict[str, Any],
        lora_paths: list[str] | None = None,
    ) -> Sampling:
        parsed = SamplingConfig.from_dict(config)
        self._validate_output_dir(parsed)
        entity_lora_paths = (
            lora_paths if lora_paths is not None else resolve_lora_paths_from_sampling_config(parsed)
        )
        parsed, paths = prepare_sampling_config_lora_paths(parsed, entity_lora_paths or None)
        if paths:
            validate_lora_paths(paths)
        validate_sample_prompts(parsed)
        snapshot = parsed.with_resolved_gpu(settings.gpu_defaults)
        snapshot.validate_gpu()
        sampling = Sampling(
            name=name,
            config=snapshot.to_snapshot(),
            lora_paths=paths,
            status=RunnableStatus.DRAFT,
        )
        sampling = await self._repo.add(sampling)
        output_dir = resolve_sampling_output_dir(snapshot, sampling.id)  # type: ignore[arg-type]
        sampling.output_path = str(output_dir)
        await self._repo.save_and_flush(sampling)
        return sampling

    async def enqueue_sampling(self, sampling_id: int) -> Sampling:
        sampling = await self.get_sampling(sampling_id)

        async def before_enqueue() -> None:
            get_runnable_handler("sampling").validate_config(sampling.config or {})
            sampling.progress_step = None
            sampling.progress_total = None
            sampling.progress_status = None

        await self._repo.enqueue_runnable(
            sampling,
            kind="Sampling",
            entity_id=sampling_id,
            before_enqueue=before_enqueue,
        )
        return sampling

    async def cancel_sampling(self, sampling_id: int) -> Sampling:
        sampling = await self.get_sampling(sampling_id)
        return await self._repo.cancel_runnable(
            sampling,
            kind="Sampling",
            entity_id=sampling_id,
        )

    async def get_logs(self, sampling_id: int, tail: int = 500) -> list[str]:
        sampling = await self.get_sampling(sampling_id)
        return read_runnable_logs(sampling, tail)

    def get_lora_paths(self, sampling: Sampling) -> list[str]:
        return [str(path) for path in (sampling.lora_paths or [])]

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

    def live_preview_path(self, sampling: Sampling) -> Path:
        if not sampling.output_path:
            raise LivePreviewNotReadyError(sampling.id or 0)
        target = resolve_safe_sample_file(Path(sampling.output_path), LIVE_PREVIEW_FILENAME)
        if target is None:
            raise LivePreviewNotReadyError(sampling.id or 0)
        return target
