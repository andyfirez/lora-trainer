"""Tests for grid planner."""

from src.sampler.experiment.grid_planner import plan_grids
from src.sampler.experiment.models import (
    GridLayout,
    ExperimentMode,
    ExperimentParameter,
    ExperimentParameters,
)


def test_two_vary_params_single_grid() -> None:
    params = ExperimentParameters(
        prompt=ExperimentParameter(mode=ExperimentMode.VARY, values=["p1", "p2"]),
        lora_weight=ExperimentParameter(mode=ExperimentMode.VARY, values=[0.5, 1.0]),
    )
    plans = plan_grids(params, GridLayout(x_axis="lora_weight", y_axis="prompt"))
    assert len(plans) == 1
    assert plans[0].cells == [[0, 2], [1, 3]]


def test_three_vary_params_multiple_grids() -> None:
    params = ExperimentParameters(
        prompt=ExperimentParameter(mode=ExperimentMode.VARY, values=["p1", "p2"]),
        lora_weight=ExperimentParameter(mode=ExperimentMode.VARY, values=[0.5, 1.0]),
        steps=ExperimentParameter(mode=ExperimentMode.VARY, values=[20, 30]),
    )
    plans = plan_grids(
        params,
        GridLayout(x_axis="lora_weight", y_axis="prompt"),
    )
    assert len(plans) == 2
