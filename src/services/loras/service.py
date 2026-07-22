"""Business logic for trained LoRA catalog."""

from typing import Sequence

from src.db.repositories.job_repo import JobRepository
from src.db.repositories.trained_lora_repo import TrainedLoraRepository
from src.db.tables.job import Job, JobStatus, JobType
from src.db.tables.trained_lora import TrainedLora
from src.services.loras.exceptions import TrainedLoraNotFoundError
from src.services.loras.paths import (
    assign_unique_training_job_yaml,
    resolve_trained_lora_paths,
)


class TrainedLoraService:
    def __init__(
        self,
        lora_repo: TrainedLoraRepository,
        job_repo: JobRepository,
    ) -> None:
        self._lora_repo = lora_repo
        self._job_repo = job_repo

    async def list_loras(self) -> Sequence[TrainedLora]:
        return await self._lora_repo.list_all()

    async def get_lora(self, lora_id: int) -> TrainedLora:
        lora = await self._lora_repo.get_by_id(lora_id)
        if lora is None:
            raise TrainedLoraNotFoundError(lora_id)
        return lora

    async def get_by_job_id(self, job_id: int) -> TrainedLora | None:
        return await self._lora_repo.get_by_job_id(job_id)

    async def create_from_completed_job(self, job: Job) -> TrainedLora | None:
        if job.id is None or job.job_type != JobType.TRAINING or job.status != JobStatus.COMPLETED:
            return None
        existing = await self._lora_repo.get_by_job_id(job.id)
        if existing is not None:
            return existing
        paths = resolve_trained_lora_paths(job)
        if paths is None:
            return None
        lora = TrainedLora(
            name=paths.name,
            job_id=job.id,
            config_id=job.config_id,
            config_yaml=job.config_yaml,
            base_model_name=paths.base_model_name,
            weights_path=str(paths.weights_path),
            work_dir=str(paths.work_dir),
        )
        return await self._lora_repo.add(lora)

    async def reproduce(
        self,
        lora_id: int,
        *,
        name: str | None = None,
    ) -> Job:
        lora = await self.get_lora(lora_id)
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
