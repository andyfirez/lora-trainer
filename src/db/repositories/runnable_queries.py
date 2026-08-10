"""Cross-entity read helpers over the Lora and Sampling tables.

These are the only places that know about *both* Runnable tables at once —
used by the global FIFO queue and the worker poll loop. Domain-specific
queries belong on `LoraRepository` / `SamplingRepository` instead.
"""

from typing import Literal, Optional, Union

from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.tables.lora import Lora
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling

RunnableKind = Literal["lora", "sampling"]

RUNNABLE_TABLES: dict[RunnableKind, type[Lora] | type[Sampling]] = {
    "lora": Lora,
    "sampling": Sampling,
}


async def get_by_kind(session: AsyncSession, kind: RunnableKind, entity_id: int) -> Union[Lora, Sampling, None]:
    return await session.get(RUNNABLE_TABLES[kind], entity_id)


async def max_queue_position(session: AsyncSession) -> int:
    positions = []
    for model in RUNNABLE_TABLES.values():
        result = await session.exec(select(func.max(model.queue_position)))
        positions.append(result.first() or 0)
    return max(positions)


async def next_queued(session: AsyncSession) -> Optional[tuple[RunnableKind, int]]:
    """Return (kind, id) of the queued entity with the smallest queue_position across both tables."""
    candidates: list[tuple[int, RunnableKind, int]] = []
    for kind, model in RUNNABLE_TABLES.items():
        result = await session.exec(
            select(model)
            .where(model.status == RunnableStatus.QUEUED)
            .order_by(model.queue_position)
            .limit(1)
        )
        entity = result.first()
        if entity is not None and entity.id is not None:
            candidates.append((entity.queue_position or 0, kind, entity.id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, kind, entity_id = candidates[0]
    return kind, entity_id


async def running_any(session: AsyncSession) -> Optional[tuple[RunnableKind, int]]:
    for kind, model in RUNNABLE_TABLES.items():
        result = await session.exec(select(model).where(model.status == RunnableStatus.RUNNING).limit(1))
        entity = result.first()
        if entity is not None and entity.id is not None:
            return kind, entity.id
    return None


async def count_running(session: AsyncSession) -> int:
    total = 0
    for model in RUNNABLE_TABLES.values():
        result = await session.exec(
            select(func.count()).select_from(model).where(model.status == RunnableStatus.RUNNING)
        )
        total += result.first() or 0
    return total
