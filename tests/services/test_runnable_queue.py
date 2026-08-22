"""Tests for the shared runnable queue/runtime helpers across Lora and Sampling."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.repositories.runnable_queries import next_queued, running_any
from src.db.tables.lora import Lora
from src.db.tables.runnable_mixin import RunnableStatus
from src.db.tables.sampling import Sampling
from src.services.runnable import queue, runtime


@pytest.mark.asyncio
async def test_enqueue_assigns_incrementing_positions_across_both_tables(session: AsyncSession) -> None:
    lora = Lora(name="lora-1", config_yaml="base_model_name: x")
    sampling = Sampling(name="sampling-1", config={"output_dir": "/tmp"})
    session.add(lora)
    session.add(sampling)
    await session.flush()

    await queue.enqueue(session, lora)
    await queue.enqueue(session, sampling)

    assert lora.queue_position == 1
    assert sampling.queue_position == 2
    assert lora.status == RunnableStatus.QUEUED
    assert sampling.status == RunnableStatus.QUEUED


@pytest.mark.asyncio
async def test_next_queued_returns_lowest_position_across_tables(session: AsyncSession) -> None:
    lora = Lora(name="lora-2", config_yaml="base_model_name: x")
    sampling = Sampling(name="sampling-2", config={"output_dir": "/tmp"})
    session.add(lora)
    session.add(sampling)
    await session.flush()

    await queue.enqueue(session, sampling)
    await queue.enqueue(session, lora)

    kind, entity_id = await next_queued(session)
    assert (kind, entity_id) == ("sampling", sampling.id)


@pytest.mark.asyncio
async def test_running_any_finds_running_entity_of_either_kind(session: AsyncSession) -> None:
    sampling = Sampling(name="sampling-3", config={"output_dir": "/tmp"}, status=RunnableStatus.RUNNING)
    session.add(sampling)
    await session.flush()

    kind, entity_id = await running_any(session)
    assert (kind, entity_id) == ("sampling", sampling.id)


def test_mark_running_then_mark_finished_accumulates_elapsed_seconds() -> None:
    lora = Lora(name="runtime-lora", config_yaml="base_model_name: x")
    runtime.mark_running(lora, pid=1234)
    assert lora.status == RunnableStatus.RUNNING
    assert lora.pid == 1234

    lora.running_started_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    runtime.mark_finished(lora, RunnableStatus.COMPLETED)

    assert lora.status == RunnableStatus.COMPLETED
    assert lora.pid is None
    assert lora.error_message is None
    assert lora.accumulated_elapsed_seconds >= 10


def test_cancel_removes_queue_position_and_sets_status() -> None:
    lora = Lora(name="cancel-lora", config_yaml="base_model_name: x", status=RunnableStatus.QUEUED, queue_position=3)
    runtime.cancel(lora)
    assert lora.status == RunnableStatus.CANCELLED
    assert lora.queue_position is None


def test_compute_elapsed_seconds_while_running() -> None:
    lora = Lora(
        name="elapsed-lora",
        config_yaml="base_model_name: x",
        status=RunnableStatus.RUNNING,
        running_started_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    )
    elapsed = runtime.compute_elapsed_seconds(lora)
    assert elapsed is not None
    assert elapsed >= 5


def test_compute_elapsed_seconds_returns_none_when_never_run() -> None:
    lora = Lora(name="never-run", config_yaml="base_model_name: x")
    assert runtime.compute_elapsed_seconds(lora) is None
