"""Tests for prompt composition helpers."""

import pytest

from src.db.tables.prompt_builder import PromptSectionValue, PromptSelectionMode, PromptTemplateSection
from src.services.prompt_builder.compose import (
    compose_from_selected_texts,
    compose_prompt,
    compose_section,
    compose_template_prompt,
    validate_section_value_text,
)


def test_compose_section_random() -> None:
    assert compose_section(["red", "blue"], mode=PromptSelectionMode.RANDOM, selected_text=None) == "{red|blue}"


def test_compose_section_fixed() -> None:
    assert compose_section(["red", "blue"], mode=PromptSelectionMode.FIXED, selected_text="red") == "red"


def test_compose_from_selected_texts_single() -> None:
    assert compose_from_selected_texts(["red"]) == "red"


def test_compose_section_fixed_multiple() -> None:
    assert compose_section(["red", "blue"], mode=PromptSelectionMode.FIXED, selected_texts=["red", "blue"]) == "red, blue"


def test_compose_fixed_texts_multiple() -> None:
    from src.services.prompt_builder.compose import compose_fixed_texts

    assert compose_fixed_texts(["red", "blue"]) == "red, blue"


def test_compose_prompt_skips_empty_segments() -> None:
    assert compose_prompt(["1girl", "", "masterpiece"]) == "1girl, masterpiece"


def test_validate_section_value_text_rejects_wildcard_chars() -> None:
    with pytest.raises(ValueError):
        validate_section_value_text("bad|value")


def test_compose_random_texts_include_empty() -> None:
    from src.services.prompt_builder.compose import compose_random_texts

    assert compose_random_texts(["soft", "dramatic"], include_empty=True) == "{|soft|dramatic}"
    assert compose_random_texts(["soft"], include_empty=True) == "{|soft}"
    assert compose_random_texts([], include_empty=True) == "{|}"


def test_compose_template_prompt_random_include_empty() -> None:
    links = [
        PromptTemplateSection(
            id=1,
            template_id=1,
            section_id=10,
            sort_order=0,
            selection_mode=PromptSelectionMode.RANDOM,
            selected_value_ids=[100, 101],
            include_empty=True,
        ),
    ]
    values_by_section = {
        10: [
            PromptSectionValue(id=100, section_id=10, text="soft", sort_order=0),
            PromptSectionValue(id=101, section_id=10, text="dramatic", sort_order=1),
        ],
    }
    assert compose_template_prompt(links, values_by_section) == "{|soft|dramatic}"


def test_compose_template_prompt() -> None:
    links = [
        PromptTemplateSection(
            id=1,
            template_id=1,
            section_id=10,
            sort_order=0,
            selection_mode=PromptSelectionMode.FIXED,
            selected_value_id=100,
            selected_value_ids=[100],
        ),
        PromptTemplateSection(
            id=2,
            template_id=1,
            section_id=20,
            sort_order=1,
            selection_mode=PromptSelectionMode.FIXED,
            selected_value_id=200,
            selected_value_ids=[200, 201],
        ),
    ]
    values_by_section = {
        10: [PromptSectionValue(id=100, section_id=10, text="1girl", sort_order=0)],
        20: [
            PromptSectionValue(id=200, section_id=20, text="anime", sort_order=0),
            PromptSectionValue(id=201, section_id=20, text="realistic", sort_order=1),
        ],
    }
    assert compose_template_prompt(links, values_by_section) == "1girl, anime, realistic"
