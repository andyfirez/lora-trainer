"""Lora runnable handler — training subprocess dispatch + completion."""

import sys

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.loras.service import LoraService
from src.services.runnable import runtime
from src.services.runnable.handlers.base import RunnableHandler
from src.trainer.config import TrainConfig


class LoraHandler(RunnableHandler):
    def build_command(self, entity_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.trainer.runner", "--lora-id", str(entity_id)]

    def validate_config_yaml(self, config_yaml: str) -> None:
        config = TrainConfig.from_snapshot_yaml(config_yaml)
        config.validate_gpu()
        if not config.concepts:
            raise ValueError("At least one training concept is required")
        for concept in config.concepts:
            if concept.dataset_id <= 0:
                raise ValueError(f"Invalid dataset_id: {concept.dataset_id}")

    async def finalize(
        self,
        session: AsyncSession,
        entity_id: int,
        exit_code: int,
        *,
        error_message: str | None = None,
    ) -> None:
        repo = LoraRepository(session)
        lora = await repo.get_by_id(entity_id)
        if lora is None or lora.status != RunnableStatus.RUNNING:
            return
        final_status = RunnableStatus.COMPLETED if exit_code == 0 else RunnableStatus.FAILED
        runtime.mark_finished(
            lora,
            final_status,
            error_message=(lora.error_message or error_message) if final_status == RunnableStatus.FAILED else None,
        )
        session.add(lora)
        await session.flush()
        if final_status == RunnableStatus.COMPLETED:
            await LoraService(repo, DatasetRepository(session)).finalize_completed_training(lora)
