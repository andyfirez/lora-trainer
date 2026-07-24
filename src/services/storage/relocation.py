"""Shared helpers for matching relocated catalog items on disk."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def unique_match(candidates: list[T]) -> T | None:
    if len(candidates) == 1:
        return candidates[0]
    return None


def folder_basename(relative_path: str) -> str:
    return Path(relative_path.replace("\\", "/")).name


def match_by_basename(
    stale_items: list[T],
    *,
    get_relative_path: Callable[[T], str],
    discovered_basename: str,
) -> list[T]:
    return [
        item
        for item in stale_items
        if folder_basename(get_relative_path(item)) == discovered_basename
    ]
