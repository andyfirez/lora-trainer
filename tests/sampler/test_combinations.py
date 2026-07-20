"""Tests for sweep combination builder."""

from src.sampler.sweep.combinations import build_combinations, count_combinations
from src.sampler.sweep.models import SweepMode, SweepParameter, SweepParameters


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
    assert combos[0].params["lora_path"] is None
    assert combos[0].params["lora_trigger"] == ""


def test_vary_prompt_and_weight() -> None:
    params = _params(
        prompt=SweepParameter(mode=SweepMode.VARY, values=["a", "b"]),
        lora_weight=SweepParameter(mode=SweepMode.VARY, values=[0.5, 1.0]),
    )
    assert count_combinations(params) == 4
    combos = build_combinations(params)
    assert len(combos) == 4


def test_vary_lora_path_dedupes_multiple_empty(tmp_path) -> None:
    lora_path = tmp_path / "demo.safetensors"
    params = _params(
        prompt=SweepParameter(mode=SweepMode.FIXED, value="hello"),
        lora_path=SweepParameter(
            mode=SweepMode.VARY,
            values=[
                {"path": "", "trigger": ""},
                {"path": None, "trigger": ""},
                {"path": str(lora_path), "trigger": "ohwx"},
            ],
        ),
    )
    combos = build_combinations(params)
    assert len(combos) == 2
    triggers = {combo.params.get("lora_trigger") for combo in combos}
    assert triggers == {"", "ohwx"}


def test_vary_lora_path_pairs_triggers(tmp_path) -> None:
    lora_a = tmp_path / "a.safetensors"
    lora_b = tmp_path / "b.safetensors"
    params = _params(
        prompt=SweepParameter(mode=SweepMode.FIXED, value="portrait"),
        lora_path=SweepParameter(
            mode=SweepMode.VARY,
            values=[
                {"path": str(lora_a), "trigger": "ohwx"},
                {"path": str(lora_b), "trigger": "sks"},
            ],
        ),
    )
    combos = build_combinations(params)
    assert len(combos) == 2
    by_path = {combo.params["lora_path"]: combo.params["lora_trigger"] for combo in combos}
    assert by_path[str(lora_a)] == "ohwx"
    assert by_path[str(lora_b)] == "sks"


def test_empty_prompt_returns_no_combinations() -> None:
    params = _params(prompt=SweepParameter(mode=SweepMode.FIXED, value=""))
    assert build_combinations(params) == []
