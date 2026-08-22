"""Prompt assembly helpers."""

from src.db.tables.prompt_builder import (
    PromptSectionValue,
    PromptSelectionMode,
    PromptTemplateSection,
)

PROMPT_SEGMENT_SEPARATOR = ", "

FORBIDDEN_VALUE_CHARS = frozenset("{}|")


def validate_section_value_text(text: str) -> None:
    if any(char in text for char in FORBIDDEN_VALUE_CHARS):
        raise ValueError("Value text cannot contain '{', '}', or '|'")


def resolve_selected_value_ids(link: PromptTemplateSection) -> list[int]:
    if link.selected_value_ids:
        return list(link.selected_value_ids)
    if link.selected_value_id is not None:
        return [link.selected_value_id]
    return []


def compose_selected_texts(
    section_values: list[PromptSectionValue],
    selected_ids: list[int],
) -> list[str]:
    selected_set = set(selected_ids)
    return [
        value.text.strip()
        for value in section_values
        if value.id in selected_set and value.text.strip()
    ]


def compose_from_selected_texts(texts: list[str]) -> str:
    cleaned = [text.strip() for text in texts if text.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return "{" + "|".join(cleaned) + "}"


def compose_fixed_texts(texts: list[str]) -> str:
    cleaned = [text.strip() for text in texts if text.strip()]
    return PROMPT_SEGMENT_SEPARATOR.join(cleaned)


def compose_random_texts(texts: list[str], *, include_empty: bool = False) -> str:
    cleaned = [text.strip() for text in texts if text.strip()]
    if include_empty:
        if not cleaned:
            return "{|}"
        return "{" + "|".join([""] + cleaned) + "}"
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return "{" + "|".join(cleaned) + "}"


def compose_section(
    values: list[str],
    *,
    mode: PromptSelectionMode,
    selected_texts: list[str] | None = None,
    selected_text: str | None = None,
) -> str:
    if mode == PromptSelectionMode.RANDOM:
        cleaned = [value.strip() for value in values if value.strip()]
        return compose_random_texts(cleaned, include_empty=False)
    if selected_texts is not None:
        return compose_fixed_texts(selected_texts)
    return (selected_text or "").strip()


def compose_prompt(segments: list[str], separator: str = PROMPT_SEGMENT_SEPARATOR) -> str:
    return separator.join(segment for segment in segments if segment.strip())


def compose_template_prompt(
    links: list[PromptTemplateSection],
    values_by_section: dict[int, list[PromptSectionValue]],
) -> str:
    segments: list[str] = []
    for link in links:
        section_values = values_by_section.get(link.section_id, [])
        selected_ids = resolve_selected_value_ids(link)
        if link.selection_mode == PromptSelectionMode.RANDOM:
            if not selected_ids:
                texts = [value.text for value in section_values]
            else:
                texts = compose_selected_texts(section_values, selected_ids)
            segments.append(compose_random_texts(texts, include_empty=link.include_empty))
        else:
            texts = compose_selected_texts(section_values, selected_ids)
            segments.append(compose_fixed_texts(texts))
    return compose_prompt(segments)
