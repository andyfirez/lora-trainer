from pathlib import Path

import pytest

from src.trainer.sdxl.dataset import count_latent_cache_items, count_te_cache_items
from src.trainer.training_log import JobTrainingLogger, LossRecorder, build_tensorboard_dir, format_step_log, setup_tensorboard_writer


def test_loss_recorder_moving_average() -> None:
    recorder = LossRecorder()
    recorder.add(epoch=0, step=0, loss=0.2)
    recorder.add(epoch=0, step=1, loss=0.4)
    assert recorder.moving_average == pytest.approx(0.3)


def test_loss_recorder_replaces_step_on_later_epoch() -> None:
    recorder = LossRecorder()
    recorder.add(epoch=0, step=0, loss=0.2)
    recorder.add(epoch=1, step=0, loss=0.8)
    assert recorder.moving_average == pytest.approx(0.8)


def test_format_step_log() -> None:
    line = format_step_log(
        step=50,
        total_steps=500,
        loss=0.0912,
        avr_loss=0.0842,
        lr=1e-4,
        epoch=1,
        epoch_total=10,
    )
    assert "step 50/500" in line
    assert "loss=0.0912" in line
    assert "avr_loss=0.0842" in line
    assert "lr=1.00e-04" in line
    assert "epoch=1/10" in line


def test_count_cache_items() -> None:
    paths = [Path("a.jpg"), Path("b.jpg"), Path("a.jpg")]
    pairs = [(Path("a.jpg"), "cat"), (Path("b.jpg"), "dog"), (Path("c.jpg"), "cat")]
    assert count_latent_cache_items(paths) == 2
    assert count_te_cache_items(pairs) == 2


def test_log_cache_progress_writes_log(tmp_path) -> None:
    logger = JobTrainingLogger(job_id=9101, log_path=tmp_path / "job.log")
    logger.create_progress_bar(10, desc="cache latents")
    logger.log_cache_progress("latents", 1, 5)
    logger.advance_progress(1, desc="cache latents")
    for handler in logger.logger.handlers:
        handler.flush()
    lines = JobTrainingLogger.read_tail(logger.log_path)
    assert any("cache latents 1/5" in line for line in lines)


def test_build_latent_cache_uses_job_logger(tmp_path) -> None:
    from unittest.mock import MagicMock

    import torch

    from src.trainer.sdxl.latent_cache import build_latent_cache

    job_logger = JobTrainingLogger(job_id=9102, log_path=tmp_path / "job.log")
    vae = MagicMock()
    vae.eval = MagicMock()
    vae.to = MagicMock(return_value=vae)

    build_latent_cache([], vae, torch.device("cpu"), False, log=job_logger.logger)
    for handler in job_logger.logger.handlers:
        handler.flush()

    lines = JobTrainingLogger.read_tail(job_logger.log_path)
    assert any("Caching latents for 0 unique images" in line for line in lines)
    assert any("Latent cache ready" in line for line in lines)


def test_build_tensorboard_dir_uses_job_id(tmp_path) -> None:
    assert build_tensorboard_dir(str(tmp_path), 42) == tmp_path / "job_42"


def test_setup_tensorboard_writer_resets_existing_run(tmp_path) -> None:
    log_dir = tmp_path / "tb"
    summary_dir = build_tensorboard_dir(str(log_dir), 7)
    summary_dir.mkdir(parents=True)
    stale_file = summary_dir / "stale.txt"
    stale_file.write_text("old run", encoding="utf-8")

    writer, created_dir = setup_tensorboard_writer(str(log_dir), 7)

    try:
        assert created_dir == summary_dir
        assert not stale_file.exists()
        writer.add_scalar("loss", 0.5, 1)
        writer.flush()
        assert any(summary_dir.iterdir())
    finally:
        writer.close()
