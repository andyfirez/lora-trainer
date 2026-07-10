from pathlib import Path
from unittest.mock import MagicMock

from src.trainer.metric_logger import MetricLogger
from src.trainer.training_log import JobTrainingLogger


def test_log_step_writes_metrics_at_log_every(tmp_path: Path) -> None:
    metric_logger = MagicMock(spec=MetricLogger)
    logger = JobTrainingLogger(
        job_id=1,
        log_path=tmp_path / "job.log",
        metric_logger=metric_logger,
        log_every=2,
    )

    logger.log_step(
        step=2,
        total_steps=10,
        loss=0.5,
        lr=1e-4,
        epoch=1,
        epoch_total=1,
        epoch_step=1,
    )

    metric_logger.log.assert_called_once_with(
        {
            "loss/loss": 0.5,
            "loss/avr_loss": 0.5,
            "learning_rate": 1e-4,
        }
    )
    metric_logger.commit.assert_called_once_with(step=2)


def test_log_step_skips_metric_log_between_intervals(tmp_path: Path) -> None:
    metric_logger = MagicMock(spec=MetricLogger)
    logger = JobTrainingLogger(
        job_id=1,
        log_path=tmp_path / "job.log",
        metric_logger=metric_logger,
        log_every=2,
    )

    logger.log_step(
        step=1,
        total_steps=10,
        loss=0.5,
        lr=1e-4,
        epoch=1,
        epoch_total=1,
        epoch_step=0,
    )

    metric_logger.log.assert_not_called()
    metric_logger.commit.assert_called_once_with(step=1)


def test_finish_closes_metric_logger(tmp_path: Path) -> None:
    metric_logger = MagicMock(spec=MetricLogger)
    logger = JobTrainingLogger(
        job_id=1,
        log_path=tmp_path / "job.log",
        metric_logger=metric_logger,
    )
    logger.finish()
    metric_logger.finish.assert_called_once()
