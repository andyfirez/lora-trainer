"""Training logging utilities — kohya-style step logs and loss tracking."""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.trainer.config import TrainConfig


class LossRecorder:
    def __init__(self) -> None:
        self._loss_list: list[float] = []
        self._loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self._loss_list.append(loss)
        else:
            while len(self._loss_list) <= step:
                self._loss_list.append(0.0)
            self._loss_total -= self._loss_list[step]
            self._loss_list[step] = loss
        self._loss_total += loss

    @property
    def moving_average(self) -> float:
        if not self._loss_list:
            return 0.0
        return self._loss_total / len(self._loss_list)


def format_step_log(
    *,
    step: int,
    total_steps: int,
    loss: float,
    avr_loss: float,
    lr: float,
    epoch: int,
    epoch_total: int,
) -> str:
    return (
        f"step {step}/{total_steps} | loss={loss:.4f} avr_loss={avr_loss:.4f} "
        f"lr={lr:.2e} epoch={epoch}/{epoch_total}"
    )


@dataclass
class JobTrainingLogger:
    job_id: int
    log_path: Path
    _loss_recorder: LossRecorder = field(default_factory=LossRecorder)
    _progress_bar: Optional[tqdm] = None
    _logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"trainer.job.{self.job_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            self._logger.addHandler(stream_handler)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def loss_recorder(self) -> LossRecorder:
        return self._loss_recorder

    def log_training_start(
        self,
        config: TrainConfig,
        *,
        epochs: int,
        steps_per_epoch: int,
        total_steps: int,
    ) -> None:
        self._logger.info(
            "Starting training: %d epochs, %d steps/epoch, total %d steps",
            epochs,
            steps_per_epoch,
            total_steps,
        )
        self._logger.info(
            "batch_size=%d, lr=%.2e, resolution=%d, gradient_accumulation_steps=%d",
            config.batch_size,
            config.learning_rate,
            config.resolution,
            config.gradient_accumulation_steps,
        )

    def log_epoch(self, epoch: int, epoch_total: int) -> None:
        self._logger.info("epoch %d/%d", epoch, epoch_total)

    def create_progress_bar(self, total_steps: int, initial: int = 0, desc: str = "steps") -> tqdm:
        self.close_progress_bar()
        self._progress_bar = tqdm(
            total=total_steps,
            initial=initial,
            desc=desc,
            smoothing=0,
            file=sys.stdout,
        )
        return self._progress_bar

    def advance_progress(self, n: int = 1, desc: str | None = None) -> int:
        if self._progress_bar is None:
            return 0
        if desc is not None:
            self._progress_bar.set_description(desc)
        self._progress_bar.update(n)
        return int(self._progress_bar.n)

    def log_cache_progress(self, phase: str, current: int, phase_total: int) -> None:
        self._logger.info("cache %s %d/%d", phase, current, phase_total)
        if self._progress_bar is not None:
            self._progress_bar.set_description(f"cache {phase}")
            self._progress_bar.set_postfix(phase=f"{current}/{phase_total}")

    def log_step(
        self,
        *,
        step: int,
        total_steps: int,
        loss: float,
        lr: float,
        epoch: int,
        epoch_total: int,
        epoch_step: int,
    ) -> float:
        self._loss_recorder.add(epoch=epoch - 1, step=epoch_step, loss=loss)
        avr_loss = self._loss_recorder.moving_average
        if self._progress_bar is not None:
            self._progress_bar.set_postfix(avr_loss=f"{avr_loss:.4f}", lr=f"{lr:.2e}")
            self._progress_bar.update(1)
        self._logger.info(
            format_step_log(
                step=step,
                total_steps=total_steps,
                loss=loss,
                avr_loss=avr_loss,
                lr=lr,
                epoch=epoch,
                epoch_total=epoch_total,
            )
        )
        return avr_loss

    def close_progress_bar(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    @staticmethod
    def read_tail(log_path: Path, lines: int = 500) -> list[str]:
        if not log_path.is_file():
            return []
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return content[-lines:]
