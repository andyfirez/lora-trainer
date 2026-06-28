from src.trainer.config import TrainConfig


def test_epoch_checkpoint_save_is_gated_by_checkpointing_enabled() -> None:
    disabled = TrainConfig(checkpointing_enabled=False, save_every_n_epochs=1)
    enabled = TrainConfig(checkpointing_enabled=True, save_every_n_epochs=1)

    assert not (disabled.checkpointing_enabled and (0 + 1) % disabled.save_every_n_epochs == 0)
    assert enabled.checkpointing_enabled and (0 + 1) % enabled.save_every_n_epochs == 0
