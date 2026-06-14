from pathlib import Path

from src.trainer.sdxl.checkpoint_state import (
    find_latest_checkpoint,
    load_resume_state,
    parse_epoch_from_checkpoint_name,
    parse_step_from_checkpoint_name,
    save_resume_state,
)


def test_save_and_load_resume_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lora_epoch2.safetensors"
    checkpoint.write_bytes(b"x")
    save_resume_state(
        checkpoint_path=checkpoint,
        lora_state_dict={"a": 1},
        optimizer_state_dict={"b": 2},
        lr_scheduler_state_dict={"c": 3},
        epoch=2,
        global_step=120,
        epoch_step=5,
    )

    state = load_resume_state(checkpoint)
    assert state.epoch == 2
    assert state.global_step == 120
    assert state.epoch_step == 5
    assert state.lora_state_dict["a"] == 1


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    older = tmp_path / "demo_epoch1.safetensors"
    newer = tmp_path / "demo_step50.safetensors"
    older.write_bytes(b"1")
    newer.write_bytes(b"2")

    latest = find_latest_checkpoint(tmp_path, "demo", "safetensors")
    assert latest in {older, newer}


def test_parse_epoch_and_step_from_name() -> None:
    assert parse_epoch_from_checkpoint_name(Path("x_epoch12.safetensors")) == 12
    assert parse_step_from_checkpoint_name(Path("x_step77.safetensors")) == 77
