"""Tests for LoraHandler.finalize success/failure paths."""

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.tables.runnable_mixin import RunnableStatus
from src.services.loras.service import LoraService
from src.services.runnable import runtime
from src.services.runnable.handlers.lora import LoraHandler


@pytest.mark.asyncio
async def test_finalize_success_marks_completed_and_sets_paths(
    session: AsyncSession,
    lora_service: LoraService,
    minimal_training_yaml: str,
    storage_roots,
) -> None:
    lora = await lora_service.create_lora(name="finalize-ok", config_yaml=minimal_training_yaml)
    work_dir = storage_roots["lora"] / "finalize-ok"
    work_dir.mkdir()
    (work_dir / "finalize-ok.safetensors").write_bytes(b"fake-weights")

    runtime.mark_running(lora, pid=12345)
    lora.output_path = str(work_dir)
    session.add(lora)
    await session.flush()

    await LoraHandler().finalize(session, lora.id, 0)
    await session.commit()

    refreshed = await lora_service.get_lora(lora.id)
    assert refreshed.status == RunnableStatus.COMPLETED
    assert refreshed.pid is None
    assert refreshed.relative_path == "finalize-ok"
    assert refreshed.weights_relpath == "finalize-ok/finalize-ok.safetensors"
    assert refreshed.error_message is None


@pytest.mark.asyncio
async def test_finalize_failure_marks_failed_without_paths(
    session: AsyncSession,
    lora_service: LoraService,
    minimal_training_yaml: str,
) -> None:
    lora = await lora_service.create_lora(name="finalize-fail", config_yaml=minimal_training_yaml)
    runtime.mark_running(lora, pid=999)
    session.add(lora)
    await session.flush()

    await LoraHandler().finalize(session, lora.id, 1, error_message="boom")
    await session.commit()

    refreshed = await lora_service._repo.get_by_id(lora.id)
    assert refreshed is not None
    assert refreshed.status == RunnableStatus.FAILED
    assert refreshed.pid is None
    assert refreshed.relative_path == ""
    assert refreshed.weights_relpath == ""
    assert refreshed.error_message == "boom"
