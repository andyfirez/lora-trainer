from pathlib import Path

from src.trainer.sdxl.checkpoint_state import (
    delete_all_resume_states,
    find_latest_checkpoint,
    load_resume_state,
    parse_epoch_from_checkpoint_name,
    parse_step_from_checkpoint_name,
    prune_stale_resume_states,
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
    assert state.grad_scaler_state_dict is None


def test_save_and_load_resume_state_with_grad_scaler(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lora_step50.safetensors"
    checkpoint.write_bytes(b"x")
    scaler_state = {"scale": 512.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000}
    save_resume_state(
        checkpoint_path=checkpoint,
        lora_state_dict={"a": 1},
        optimizer_state_dict={"b": 2},
        lr_scheduler_state_dict={"c": 3},
        epoch=1,
        global_step=50,
        epoch_step=10,
        grad_scaler_state_dict=scaler_state,
    )

    state = load_resume_state(checkpoint)
    assert state.grad_scaler_state_dict == scaler_state


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


def _write_checkpoint_with_state(tmp_path: Path, name: str, epoch: int) -> Path:
    checkpoint = tmp_path / name
    checkpoint.write_bytes(b"x")
    save_resume_state(
        checkpoint_path=checkpoint,
        lora_state_dict={"epoch": epoch},
        optimizer_state_dict={"b": epoch},
        lr_scheduler_state_dict={"c": epoch},
        epoch=epoch,
        global_step=epoch * 10,
        epoch_step=0,
    )
    return checkpoint


def test_prune_stale_resume_states_keeps_latest_only(tmp_path: Path) -> None:
    first = _write_checkpoint_with_state(tmp_path, "lora_epoch1.safetensors", 1)
    second = _write_checkpoint_with_state(tmp_path, "lora_epoch2.safetensors", 2)
    keep = first.with_suffix(f"{first.suffix}.state.pt")

    prune_stale_resume_states(tmp_path, keep=keep)

    assert keep.is_file()
    assert not second.with_suffix(f"{second.suffix}.state.pt").is_file()


def test_delete_all_resume_states_removes_every_state_file(tmp_path: Path) -> None:
    _write_checkpoint_with_state(tmp_path, "lora_epoch1.safetensors", 1)
    _write_checkpoint_with_state(tmp_path, "lora_epoch2.safetensors", 2)

    delete_all_resume_states(tmp_path)

    assert list(tmp_path.glob("*.state.pt")) == []


def test_prune_then_load_resume_state_still_works(tmp_path: Path) -> None:
    _write_checkpoint_with_state(tmp_path, "lora_epoch1.safetensors", 1)
    latest = _write_checkpoint_with_state(tmp_path, "lora_epoch2.safetensors", 2)
    keep = latest.with_suffix(f"{latest.suffix}.state.pt")

    prune_stale_resume_states(tmp_path, keep=keep)

    state = load_resume_state(latest)
    assert state.epoch == 2
    assert state.lora_state_dict["epoch"] == 2
