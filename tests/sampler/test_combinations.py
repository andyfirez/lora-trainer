"""Tests for sweep combination builder."""

from src.sampler.sweep.combinations import build_combinations, count_combinations
from src.sampler.sweep.models import GridLayout, SweepMode, SweepParameter, SweepParameters


def _params(**kwargs: SweepParameter) -> SweepParameters:
    base = SweepParameters()
    return base.model_copy(update=kwargs)


def test_single_fixed_prompt() -> None:
    params = _params(
        prompt=SweepParameter(mode=SweepMode.FIXED, value="hello"),
    )
    combos = build_combinations(params)
    assert len(combos) == 1
    assert combos[0].params["prompt"] == "hello"


def test_vary_prompt_and_weight() -> None:
    params = _params(
        prompt=SweepParameter(mode=SweepMode.VARY, values=["a", "b"]),
        lora_weight=SweepParameter(mode=SweepMode.VARY, values=[0.5, 1.0]),
    )
    assert count_combinations(params) == 4
    combos = build_combinations(params)
    assert len(combos) == 4


def test_empty_prompt_returns_no_combinations() -> None:
    params = _params(prompt=SweepParameter(mode=SweepMode.FIXED, value=""))
    assert build_combinations(params) == []
