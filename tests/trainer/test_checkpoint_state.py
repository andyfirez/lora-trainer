from pathlib import Path

from src.trainer.sdxl.checkpoint_state import (
    delete_all_resume_states,
    find_latest_checkpoint,
    load_resume_state,
    parse_epoch_from_checkpoint_name,
    parse_step_from_checkpoint_name,
    prune_stale_resume_states,
    save_resume_state,
    state_path_for_checkpoint,
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


def _write_state_file(work_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = work_dir / checkpoint_name
    checkpoint.write_bytes(b"x")
    return save_resume_state(
        checkpoint_path=checkpoint,
        lora_state_dict={"a": 1},
        optimizer_state_dict={"b": 2},
        lr_scheduler_state_dict={"c": 3},
        epoch=1,
        global_step=1,
        epoch_step=0,
    )


def test_prune_stale_resume_states_keeps_latest_only(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    kept = _write_state_file(work_dir, "lora_epoch1.safetensors")
    stale = _write_state_file(work_dir, "lora_epoch2.safetensors")
    _write_state_file(work_dir, "lora_epoch3.safetensors")

    prune_stale_resume_states(work_dir, keep_state_path=kept)

    assert kept.is_file()
    assert not stale.is_file()
    assert len(list(work_dir.glob("*.state.pt"))) == 1


def test_delete_all_resume_states_preserves_checkpoints(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    checkpoint_one = work_dir / "lora_epoch1.safetensors"
    checkpoint_two = work_dir / "lora_epoch2.safetensors"
    checkpoint_one.write_bytes(b"1")
    checkpoint_two.write_bytes(b"2")
    _write_state_file(work_dir, checkpoint_one.name)
    _write_state_file(work_dir, checkpoint_two.name)

    delete_all_resume_states(work_dir)

    assert not list(work_dir.glob("*.state.pt"))
    assert checkpoint_one.is_file()
    assert checkpoint_two.is_file()


def test_save_resume_state_then_prune_allows_resume(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    first_checkpoint = work_dir / "lora_epoch1.safetensors"
    second_checkpoint = work_dir / "lora_epoch2.safetensors"
    first_checkpoint.write_bytes(b"1")
    second_checkpoint.write_bytes(b"2")

    first_state = save_resume_state(
        checkpoint_path=first_checkpoint,
        lora_state_dict={"layer": 1},
        optimizer_state_dict={"opt": 1},
        lr_scheduler_state_dict={"lr": 1},
        epoch=1,
        global_step=10,
        epoch_step=0,
    )
    second_state = save_resume_state(
        checkpoint_path=second_checkpoint,
        lora_state_dict={"layer": 2},
        optimizer_state_dict={"opt": 2},
        lr_scheduler_state_dict={"lr": 2},
        epoch=2,
        global_step=20,
        epoch_step=0,
    )
    prune_stale_resume_states(work_dir, keep_state_path=second_state)

    assert first_state == state_path_for_checkpoint(first_checkpoint)
    assert second_state == state_path_for_checkpoint(second_checkpoint)
    assert not first_state.is_file()

    resumed = load_resume_state(second_checkpoint)
    assert resumed.epoch == 2
    assert resumed.global_step == 20
    assert resumed.lora_state_dict["layer"] == 2
