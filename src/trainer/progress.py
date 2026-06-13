"""SDXL LoRA training progress tracking."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainProgress:
    global_step: int = 0
    epoch: int = 0
    epoch_step: int = 0
    epoch_total_steps: int = 0
    loss: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def next_step(self, loss: float) -> None:
        self.global_step += 1
        self.epoch_step += 1
        self.loss = loss
        self.loss_history.append(loss)

    def next_epoch(self) -> None:
        self.epoch += 1
        self.epoch_step = 0
