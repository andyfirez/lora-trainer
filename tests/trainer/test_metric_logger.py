from pathlib import Path

import pytest

from src.services.jobs.loss_log_reader import read_loss_log
from src.trainer.metric_logger import MetricLogger, build_loss_log_path
from src.trainer.config import TrainConfig


def test_build_loss_log_path() -> None:
    config = TrainConfig(output_dir="out", lora_name="my_lora")
    assert build_loss_log_path(config) == Path("out/my_lora/loss_log.db")


def test_metric_logger_log_commit_finish(tmp_path: Path) -> None:
    log_file = tmp_path / "loss_log.db"
    logger = MetricLogger(log_file)
    logger.log({"loss/loss": 0.5, "learning_rate": 1e-4})
    logger.commit(step=1)
    logger.log({"loss/loss": 0.3})
    logger.commit(step=2)
    logger.finish()

    result = read_loss_log(log_file, key="loss/loss")
    assert "loss/loss" in result.keys
    assert "learning_rate" in result.keys
    assert len(result.points) == 2
    assert result.points[0].step == 1
    assert result.points[0].value == pytest.approx(0.5)
    assert result.points[1].step == 2
    assert result.points[1].value == pytest.approx(0.3)


def test_metric_logger_prunes_future_steps_on_resume(tmp_path: Path) -> None:
    log_file = tmp_path / "loss_log.db"
    first = MetricLogger(log_file)
    for step in range(1, 6):
        first.log({"loss/loss": float(step)})
        first.commit(step=step)
    first.finish()

    second = MetricLogger(log_file)
    second.log({"loss/loss": 9.0})
    second.commit(step=3)
    second.finish()

    result = read_loss_log(log_file, key="loss/loss")
    steps = [p.step for p in result.points]
    assert steps == [1, 2, 3]
    assert result.points[-1].value == pytest.approx(9.0)


def test_read_loss_log_since_step(tmp_path: Path) -> None:
    log_file = tmp_path / "loss_log.db"
    logger = MetricLogger(log_file)
    for step in range(1, 6):
        logger.log({"loss/loss": float(step)})
        logger.commit(step=step)
    logger.finish()

    result = read_loss_log(log_file, key="loss/loss", since_step=3)
    assert [p.step for p in result.points] == [4, 5]


def test_read_loss_log_missing_file(tmp_path: Path) -> None:
    result = read_loss_log(tmp_path / "missing.db")
    assert result.keys == []
    assert result.points == []
