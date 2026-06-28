"""Caption and sample prompt assembly with trigger words."""

from src.trainer.config import ConceptConfig


def collect_trigger_words(concepts: list[ConceptConfig]) -> list[str]:
    if not concepts:
        return []
    return [word.strip() for word in concepts[0].trigger_words if word.strip()]


def join_trigger_words_and_text(trigger_words: list[str], text: str, suffix: str = "") -> str:
    parts = [word.strip() for word in trigger_words if word.strip()]
    if text.strip():
        parts.append(text.strip())
    caption = ", ".join(parts)
    return f"{caption}{suffix}"


def apply_trigger_words_to_prompt(prompt: str, trigger_words: list[str]) -> str:
    stripped = prompt.strip()
    words = [word.strip() for word in trigger_words if word.strip()]
    if not words:
        return stripped
    prefix = ", ".join(words)
    if not stripped:
        return prefix
    lower_stripped = stripped.lower()
    lower_prefix = prefix.lower()
    if lower_stripped == lower_prefix or lower_stripped.startswith(f"{lower_prefix},"):
        return stripped
    return f"{prefix}, {stripped}"


def apply_trigger_words_to_sample_prompts(
    sample_prompts: list[str],
    trigger_words: list[str],
) -> list[str]:
    if not trigger_words:
        return sample_prompts
    return [apply_trigger_words_to_prompt(prompt, trigger_words) for prompt in sample_prompts]
