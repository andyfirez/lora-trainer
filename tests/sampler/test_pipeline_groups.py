"""Tests for sweep pipeline group ordering."""

from src.sampler.sweep.combinations import build_combinations
from src.sampler.sweep.engine import sort_pipeline_groups
from src.sampler.sweep.models import SweepMode, SweepParameter, SweepParameters


def test_sort_pipeline_groups_batches_by_base_model_first() -> None:
    groups = {
        ("modelB", "/loras/a.safetensors"): [1],
        ("modelA", "/loras/b.safetensors"): [2],
        ("modelB", "/loras/b.safetensors"): [3],
        ("modelA", "/loras/a.safetensors"): [4],
    }
    sorted_keys = [key for key, _ in sort_pipeline_groups(groups)]
    assert sorted_keys == [
        ("modelA", "/loras/a.safetensors"),
        ("modelA", "/loras/b.safetensors"),
        ("modelB", "/loras/a.safetensors"),
        ("modelB", "/loras/b.safetensors"),
    ]


def test_sort_pipeline_groups_treats_none_lora_as_empty_string() -> None:
    groups = {
        ("modelB", None): [1],
        ("modelA", None): [2],
    }
    sorted_keys = [key for key, _ in sort_pipeline_groups(groups)]
    assert sorted_keys == [("modelA", None), ("modelB", None)]


def test_combinations_with_varying_base_model() -> None:
    params = SweepParameters(
        base_model_name=SweepParameter(
            mode=SweepMode.VARY,
            values=["modelA", "modelB"],
        ),
        prompt=SweepParameter(mode=SweepMode.FIXED, value="hello"),
    )
    combos = build_combinations(params)
    assert len(combos) == 2
    base_models = {combo.params["base_model_name"] for combo in combos}
    assert base_models == {"modelA", "modelB"}
    assert all(combo.params["prompt"] == "hello" for combo in combos)
