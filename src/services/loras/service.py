"""Business logic for trained LoRA catalog."""

from pathlib import Path
from typing import Sequence

from src.db.repositories.job_repo import JobRepository
from src.db.repositories.trained_lora_repo import TrainedLoraRepository
from src.db.tables.job import Job, JobStatus, JobType
from src.db.tables.trained_lora import TrainedLora
from src.services.loras.discovery import LoraDiscoveryService
from src.services.loras.exceptions import TrainedLoraNotFoundError, TrainedLoraReproduceError
from src.services.loras.paths import (
    assign_unique_training_job_yaml,
    lora_artifacts_exist,
    resolve_trained_lora_paths,
    resolve_work_dir,
)
from src.storage.paths import StorageKind, StoragePaths


class TrainedLoraService:
    def __init__(
        self,
        lora_repo: TrainedLoraRepository,
        job_repo: JobRepository,
    ) -> None:
        self._lora_repo = lora_repo
        self._job_repo = job_repo

    async def _link_jobs_to_loras(self) -> None:
        from sqlmodel import select

        from src.db.tables.job import Job

        result = await self._lora_repo._exec(
            select(TrainedLora).where(TrainedLora.job_id.is_(None))  # type: ignore[union-attr]
        )
        unlinked = result.all()
        if not unlinked:
            return

        jobs_result = await self._job_repo._exec(
            select(Job).where(
                Job.job_type == JobType.TRAINING,
                Job.status == JobStatus.COMPLETED,
                Job.output_path.is_not(None),  # type: ignore[union-attr]
            )
        )
        jobs = jobs_result.all()
        for lora in unlinked:
            try:
                work_dir = resolve_work_dir(lora).resolve()
            except (ValueError, OSError):
                continue
            for job in jobs:
                if job.id is None or job.output_path is None:
                    continue
                try:
                    output_dir = Path(job.output_path).expanduser().resolve()
                except OSError:
                    continue
                if output_dir != work_dir:
                    continue
                existing = await self._lora_repo.get_by_job_id(job.id)
                if existing is not None and existing.id != lora.id:
                    continue
                lora.job_id = job.id
                lora.config_id = job.config_id
                lora.config_yaml = job.config_yaml
                if job.config_yaml:
                    from src.services.loras.paths import runtime_train_config

                    lora.base_model_name = runtime_train_config(job).base_model_name
                self._lora_repo._session.add(lora)
                break
        await self._lora_repo._session.flush()

    async def list_loras(self) -> Sequence[TrainedLora]:
        StoragePaths.ensure_root(StorageKind.LORA)
        await self._sync_discovered_loras()
        await self._link_jobs_to_loras()
        loras = await self._lora_repo.list_all()
        return [lora for lora in loras if lora_artifacts_exist(lora)]

    async def get_lora(self, lora_id: int) -> TrainedLora:
        lora = await self._lora_repo.get_by_id(lora_id)
        if lora is None or not lora_artifacts_exist(lora):
            raise TrainedLoraNotFoundError(lora_id)
        return lora

    async def get_by_job_id(self, job_id: int) -> TrainedLora | None:
        return await self._lora_repo.get_by_job_id(job_id)

    async def _sync_discovered_loras(self) -> None:
        discovered = LoraDiscoveryService().discover_lora_work_dirs()
        existing_paths: set[str] = set()
        for lora in await self._lora_repo.list_all():
            existing_paths.add(lora.relative_path)
            canonical = StoragePaths.to_relative(StorageKind.LORA, lora.relative_path)
            if canonical is not None:
                existing_paths.add(canonical)

        for item in discovered:
            if item.relative_path in existing_paths:
                continue
            name = item.name
            candidate = name
            suffix = 2
            while await self._lora_repo.get_by_name(candidate) is not None:
                candidate = f"{name}-{suffix}"
                suffix += 1
            await self._lora_repo.add(
                TrainedLora(
                    name=candidate,
                    relative_path=item.relative_path,
                    weights_relpath=item.weights_relpath,
                    base_model_name="unknown",
                )
            )
            existing_paths.add(item.relative_path)

    async def create_from_completed_job(self, job: Job) -> TrainedLora | None:
        if job.id is None or job.job_type != JobType.TRAINING or job.status != JobStatus.COMPLETED:
            return None
        existing = await self._lora_repo.get_by_job_id(job.id)
        if existing is not None:
            return existing
        paths = resolve_trained_lora_paths(job)
        if paths is None:
            return None
        by_path = await self._lora_repo.get_by_relative_path(paths.relative_path)
        if by_path is not None:
            by_path.job_id = job.id
            by_path.config_id = job.config_id
            by_path.config_yaml = job.config_yaml
            by_path.base_model_name = paths.base_model_name
            by_path.weights_relpath = paths.weights_relpath
            by_path.name = paths.name
            self._lora_repo._session.add(by_path)
            await self._lora_repo._session.flush()
            await self._lora_repo._session.refresh(by_path)
            return by_path
        lora = TrainedLora(
            name=paths.name,
            relative_path=paths.relative_path,
            weights_relpath=paths.weights_relpath,
            job_id=job.id,
            config_id=job.config_id,
            config_yaml=job.config_yaml,
            base_model_name=paths.base_model_name,
        )
        return await self._lora_repo.add(lora)

    async def reproduce(
        self,
        lora_id: int,
        *,
        name: str | None = None,
    ) -> Job:
        lora = await self.get_lora(lora_id)
        if not lora.config_yaml:
            raise TrainedLoraReproduceError(lora_id)
        job_name = name or f"{lora.name} reproduce"
        job = Job(
            job_type=JobType.TRAINING,
            name=job_name,
            config_id=lora.config_id,
            config_yaml=lora.config_yaml,
        )
        job = await self._job_repo.add(job)
        if job.id is None:
            raise RuntimeError("Failed to create reproduction job")
        job.config_yaml = assign_unique_training_job_yaml(lora.config_yaml, job.id)
        self._job_repo._session.add(job)
        await self._job_repo._session.flush()
        await self._job_repo._session.refresh(job)
        return job
