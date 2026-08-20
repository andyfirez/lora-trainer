"""Sampling runnable handler — sampling subprocess dispatch + completion."""

import sys

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.sampling_repo import SamplingRepository
from src.db.tables.runnable_mixin import RunnableMixin, RunnableStatus
from src.db.tables.sampling import Sampling
from src.sampler.config import SamplingConfig
from src.services.runnable.handlers.base import RunnableHandler


class SamplingHandler(RunnableHandler):
    def build_command(self, entity_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.sampler.runner", "--sampling-id", str(entity_id)]

    def validate_config_yaml(self, config_yaml: str) -> None:
        config = SamplingConfig.from_snapshot_yaml(config_yaml)
        config.validate_gpu()

    async def _load_entity(self, session: AsyncSession, entity_id: int) -> Sampling | None:
        return await SamplingRepository(session).get_by_id(entity_id)

    def _before_mark_finished(self, entity: RunnableMixin, final_status: RunnableStatus) -> None:
        if final_status == RunnableStatus.COMPLETED:
            entity.progress_status = None
