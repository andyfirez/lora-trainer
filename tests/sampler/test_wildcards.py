"""Tests for wildcard expansion in sampling prompts."""

import random

from src.sampler.wildcards import expand_prompt_wildcards


def test_expand_prompt_wildcards_no_groups() -> None:
    assert expand_prompt_wildcards("plain prompt") == "plain prompt"


def test_expand_prompt_wildcards_single_group_deterministic() -> None:
    rng = random.Random(0)
    result = expand_prompt_wildcards("{red|blue|green}", rng=rng)
    assert result in {"red", "blue", "green"}
    assert expand_prompt_wildcards("{red|blue|green}", rng=random.Random(0)) == result


def test_expand_prompt_wildcards_multiple_groups() -> None:
    rng = random.Random(1)
    result = expand_prompt_wildcards("{a|b}, {1|2}", rng=rng)
    assert result.count(", ") == 1
    left, right = result.split(", ")
    assert left in {"a", "b"}
    assert right in {"1", "2"}


def test_expand_prompt_wildcards_empty_group() -> None:
    assert expand_prompt_wildcards("{|}") == ""


def test_expand_prompt_wildcards_include_empty_option() -> None:
    rng = random.Random(0)
    result = expand_prompt_wildcards("{|red|blue}", rng=rng)
    assert result in {"", "red", "blue"}
