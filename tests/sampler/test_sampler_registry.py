"""Tests for sampler/scheduler registry and legacy config mapping."""

from src.sampler.config import SamplingConfig
from src.sampler.experiment.models import (
    ExperimentMode,
    ExperimentParameter,
    ExperimentParameters,
)
from src.sampler.registry import (
    AUTOMATIC_SCHEDULER,
    SamplingProfile,
    default_scheduler_for,
    resolve_sampler_scheduler,
    schedulers_for_profile,
)
from src.sampler.registry.defaults import UnknownSamplerError, UnknownSchedulerError


def test_legacy_euler_a_splits_into_sampler_and_schedule() -> None:
    params = ExperimentParameters.model_validate(
        {"scheduler": {"mode": "fixed", "value": "euler_a"}}
    )
    assert params.sampler.first_value() == "euler_ancestral"
    assert params.scheduler.first_value() == "normal"


def test_legacy_dpm_pp_splits_to_dpmpp_2m_karras() -> None:
    config = SamplingConfig.from_dict(
        {"parameters": {"scheduler": {"mode": "fixed", "value": "dpm++"}}}
    )
    assert config.effective_sampler() == "dpmpp_2m"
    assert config.effective_scheduler() == "karras"
    updates = config.inference_config_field_updates()
    assert updates["sample_sampler"] == "dpmpp_2m"
    assert updates["sample_scheduler"] == "karras"


def test_new_sampler_scheduler_pair_is_unchanged() -> None:
    params = ExperimentParameters.model_validate(
        {
            "sampler": {"mode": "fixed", "value": "dpmpp_2m"},
            "scheduler": {"mode": "fixed", "value": "karras"},
        }
    )
    assert params.sampler.first_value() == "dpmpp_2m"
    assert params.scheduler.first_value() == "karras"


def test_automatic_scheduler_uses_per_sampler_default() -> None:
    sampler, scheduler = resolve_sampler_scheduler("dpmpp_2m", AUTOMATIC_SCHEDULER)
    assert sampler == "dpmpp_2m"
    assert scheduler == "karras"
    assert default_scheduler_for("euler") == "normal"


def test_unknown_sampler_raises() -> None:
    try:
        resolve_sampler_scheduler("not-a-sampler", "karras")
    except UnknownSamplerError:
        return
    raise AssertionError("expected UnknownSamplerError")


def test_unknown_scheduler_raises() -> None:
    try:
        resolve_sampler_scheduler("euler", "turbo")
    except UnknownSchedulerError:
        return
    raise AssertionError("expected UnknownSchedulerError")


def test_reforge_profile_includes_automatic() -> None:
    ids = [spec.id for spec in schedulers_for_profile(SamplingProfile.REFORGE)]
    assert ids[0] == AUTOMATIC_SCHEDULER
    comfy_ids = [spec.id for spec in schedulers_for_profile(SamplingProfile.COMFY)]
    assert AUTOMATIC_SCHEDULER not in comfy_ids


def test_vary_legacy_schedulers_map_both_axes() -> None:
    params = ExperimentParameters(
        scheduler=ExperimentParameter(mode=ExperimentMode.VARY, values=["euler", "euler_a"]),
    )
    assert params.sampler.mode == ExperimentMode.VARY
    assert params.sampler.values == ["euler", "euler_ancestral"]
    assert params.scheduler.values == ["normal", "normal"]
