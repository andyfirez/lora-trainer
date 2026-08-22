"""Expand {a|b|c} wildcard syntax in prompts."""

from __future__ import annotations

import random
import re

WILDCARD_RE = re.compile(r"\{([^{}]+)\}")


def expand_prompt_wildcards(prompt: str, *, rng: random.Random | None = None) -> str:
    """Replace each `{a|b|c}` group with one random alternative per call."""
    if not prompt or "{" not in prompt:
        return prompt

    source = rng or random

    def replace(match: re.Match[str]) -> str:
        body = match.group(1)
        options = [part.strip() for part in body.split("|")]
        if not options:
            return ""
        if all(option == "" for option in options):
            return ""
        return source.choice(options)

    return WILDCARD_RE.sub(replace, prompt)
