"""Lora runnable handler — training subprocess dispatch + completion."""

import sys

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.repositories.dataset_repo import DatasetRepository
from src.db.repositories.lora_repo import LoraRepository
from src.db.tables.lora import Lora
from src.db.tables.runnable_mixin import RunnableMixin
from src.services.loras.service import LoraService
from src.services.runnable.handlers.base import RunnableHandler


class LoraHandler(RunnableHandler):
    def build_command(self, entity_id: int) -> list[str]:
        return [sys.executable, "-u", "-m", "src.trainer.runner", "--lora-id", str(entity_id)]

    async def _load_entity(self, session: AsyncSession, entity_id: int) -> Lora | None:
        return await LoraRepository(session).get_by_id(entity_id)

    async def on_completed(self, session: AsyncSession, entity: RunnableMixin) -> None:
        await LoraService(LoraRepository(session), DatasetRepository(session)).finalize_completed_training(entity)
