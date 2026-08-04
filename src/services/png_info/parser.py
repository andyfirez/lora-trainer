"""Parse A1111-style generation parameter strings."""

from __future__ import annotations

import json
import re

RE_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
RE_IMAGE_SIZE = re.compile(r"^(\d+)x(\d+)$")


def _unquote(text: str) -> str:
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_generation_parameters(text: str) -> dict[str, str | int]:
    """Parse prompt, negative prompt, and comma-separated generation fields."""
    if not text.strip():
        return {}

    res: dict[str, str | int] = {}
    prompt = ""
    negative_prompt = ""
    done_with_prompt = False

    *lines, lastline = text.strip().split("\n")
    if len(RE_PARAM.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    for key, value in RE_PARAM.findall(lastline):
        if value.startswith('"') and value.endswith('"'):
            value = _unquote(value)

        size_match = RE_IMAGE_SIZE.match(value)
        if size_match is not None:
            res[f"{key}-1"] = int(size_match.group(1))
            res[f"{key}-2"] = int(size_match.group(2))
        else:
            res[key] = value

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt
    return res
