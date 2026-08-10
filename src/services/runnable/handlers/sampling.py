"""Sampling runnable handler — sampling subprocess dispatch + completion."""

import sys

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.sampling_repo import SamplingRepository
from src.db.tables.runnable_mixin import RunnableStatus
from src.sampler.config import SamplingConfig
from src.services.runnable import runtime
from src.services.runnable.handlers.base import RunnableHandler


class SamplingHandler(RunnableHandler):
    def build_command(self, entity_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.sampler.runner", "--sampling-id", str(entity_id)]

    def validate_config_yaml(self, config_yaml: str) -> None:
        config = SamplingConfig.from_snapshot_yaml(config_yaml)
        config.validate_gpu()

    async def finalize(
        self,
        session: AsyncSession,
        entity_id: int,
        exit_code: int,
        *,
        error_message: str | None = None,
    ) -> None:
        repo = SamplingRepository(session)
        sampling = await repo.get_by_id(entity_id)
        if sampling is None or sampling.status != RunnableStatus.RUNNING:
            return
        final_status = RunnableStatus.COMPLETED if exit_code == 0 else RunnableStatus.FAILED
        if final_status == RunnableStatus.COMPLETED:
            sampling.progress_status = None
        runtime.mark_finished(
            sampling,
            final_status,
            error_message=(
                (sampling.error_message or error_message) if final_status == RunnableStatus.FAILED else None
            ),
        )
        session.add(sampling)
        await session.flush()
