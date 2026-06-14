import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.migrations import migrate_schema
from src.db.repositories.queue_repo import QueueRepository
from src.db.repositories.sampling_run_repo import SamplingRunRepository
from src.db.repositories.training_job_repo import TrainingJobRepository
from src.db.tables.queue_entry import QueueItemType
from src.db.tables.sampling_run import SamplingRunStatus
from src.db.tables.training_job import JobStatus, TrainingJob
from src.services.sampling.exceptions import SamplingLoRAPathNotFoundError, SamplingPromptsNotConfiguredError
from src.services.sampling.service import SamplingService


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await migrate_schema(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session
    await engine.dispose()


@pytest_asyncio.fixture
async def sampling_service(session: AsyncSession) -> SamplingService:
    return SamplingService(
        SamplingRunRepository(session),
        QueueRepository(session),
        TrainingJobRepository(session),
    )


@pytest.mark.asyncio
async def test_create_for_job_accepts_arbitrary_lora_path(
    sampling_service: SamplingService,
    session: AsyncSession,
    tmp_path,
) -> None:
    lora_path = tmp_path / "custom.safetensors"
    lora_path.write_bytes(b"lora")
    job = TrainingJob(
        name="job",
        config_yaml=f"output_dir: {tmp_path.as_posix()}\nlora_name: demo\nsample_prompts:\n  - test prompt\n",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    sampling_run = await sampling_service.create_for_job(
        job.id,
        name=None,
        lora_paths=[str(lora_path)],
    )

    assert sampling_run.source_job_id == job.id
    assert sampling_service.get_lora_paths(sampling_run) == [str(lora_path)]
    assert sampling_run.output_path is not None


@pytest.mark.asyncio
async def test_create_run_rejects_missing_sample_prompts(sampling_service: SamplingService, tmp_path) -> None:
    lora_path = tmp_path / "model.safetensors"
    lora_path.write_bytes(b"lora")

    with pytest.raises(SamplingPromptsNotConfiguredError):
        await sampling_service.create_run(
            name=None,
            config_yaml=f"output_dir: {tmp_path.as_posix()}\nlora_name: demo\n",
            lora_paths=[str(lora_path)],
        )


@pytest.mark.asyncio
async def test_get_run_logs_tail(
    sampling_service: SamplingService,
    session: AsyncSession,
    tmp_path,
) -> None:
    log_path = tmp_path / "sampling_run_1.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    from src.db.tables.sampling_run import SamplingRun

    sampling_run = SamplingRun(
        name="sample",
        config_yaml="base_model_name: x",
        lora_paths_yaml="[]",
        log_path=str(log_path),
    )
    session.add(sampling_run)
    await session.commit()
    await session.refresh(sampling_run)

    lines = await sampling_service.get_run_logs(sampling_run.id, tail=2)

    assert lines == ["line2", "line3"]


@pytest.mark.asyncio
async def test_create_run_rejects_missing_lora_path(sampling_service: SamplingService, tmp_path) -> None:
    missing_path = tmp_path / "missing.safetensors"

    with pytest.raises(SamplingLoRAPathNotFoundError):
        await sampling_service.create_run(
            name=None,
            config_yaml="base_model_name: x\nsample_prompts:\n  - test\n",
            lora_paths=[str(missing_path)],
        )


@pytest.mark.asyncio
async def test_auto_run_enqueues_intermediate_checkpoints(
    sampling_service: SamplingService,
    session: AsyncSession,
    tmp_path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir = output_dir / "demo"
    work_dir.mkdir(parents=True)
    epoch_path = work_dir / "demo_epoch1.safetensors"
    step_path = work_dir / "demo_step10.safetensors"
    final_path = work_dir / "demo.safetensors"
    epoch_path.write_bytes(b"epoch")
    step_path.write_bytes(b"step")
    final_path.write_bytes(b"final")
    config_yaml = f"""
output_dir: {output_dir.as_posix()}
lora_name: demo
output_format: safetensors
sample_after_training: true
sample_prompts:
  - test prompt
"""
    job = TrainingJob(name="job", config_yaml=config_yaml, status=JobStatus.COMPLETED)
    session.add(job)
    await session.commit()
    await session.refresh(job)

    sampling_run = await sampling_service.create_auto_run_for_job(job)

    assert sampling_run is not None
    assert sampling_run.status == SamplingRunStatus.QUEUED
    assert sampling_service.get_lora_paths(sampling_run) == [str(epoch_path), str(step_path)]
    queue_entry = await sampling_service._queue_repo.get_by_sampling_run_id(sampling_run.id)
    assert queue_entry is not None
    assert queue_entry.item_type == QueueItemType.SAMPLING
    assert queue_entry.item_id == sampling_run.id
