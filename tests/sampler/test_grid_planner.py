"""Tests for grid planner."""

from src.sampler.sweep.grid_planner import plan_grids
from src.sampler.sweep.models import GridLayout, SweepMode, SweepParameter, SweepParameters


def test_two_vary_params_single_grid() -> None:
    params = SweepParameters(
        prompt=SweepParameter(mode=SweepMode.VARY, values=["p1", "p2"]),
        lora_weight=SweepParameter(mode=SweepMode.VARY, values=[0.5, 1.0]),
    )
    plans = plan_grids(params, GridLayout(x_axis="lora_weight", y_axis="prompt"))
    assert len(plans) == 1
    assert plans[0].cells == [[0, 2], [1, 3]]


def test_three_vary_params_multiple_grids() -> None:
    params = SweepParameters(
        prompt=SweepParameter(mode=SweepMode.VARY, values=["p1", "p2"]),
        lora_weight=SweepParameter(mode=SweepMode.VARY, values=[0.5, 1.0]),
        steps=SweepParameter(mode=SweepMode.VARY, values=[20, 30]),
    )
    plans = plan_grids(
        params,
        GridLayout(x_axis="lora_weight", y_axis="prompt"),
    )
    assert len(plans) == 2
