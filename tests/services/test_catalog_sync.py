"""Tests for generic catalog sync."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from src.services.storage.catalog_sync import sync_discovered_items


@dataclass
class _Entity:
    relative_path: str
    name: str


@dataclass
class _Discovered:
    relative_path: str
    name: str


@pytest.mark.asyncio
async def test_sync_discovered_items_creates_new_entities() -> None:
    staged: list[_Entity] = []
    created: list[tuple[_Discovered, str]] = []
    existing_paths: set[str] = set()

    async def create_entity(item: _Discovered, unique_name: str) -> None:
        created.append((item, unique_name))
        existing_paths.add(item.relative_path)

    await sync_discovered_items(
        discovered=[_Discovered("alpha", "alpha"), _Discovered("beta", "beta")],
        stale_items=[],
        existing_paths=existing_paths,
        get_discovered_path=lambda item: item.relative_path,
        find_relocated=lambda _stale, _item: None,
        apply_relocation=lambda _relocated, _item: None,
        stage=lambda entity: staged.append(entity),
        flush=lambda: _noop(),
        make_unique_name=lambda item: _async_value(item.name),
        create_entity=create_entity,
    )

    assert created == [
        (_Discovered("alpha", "alpha"), "alpha"),
        (_Discovered("beta", "beta"), "beta"),
    ]
    assert staged == []


@pytest.mark.asyncio
async def test_sync_discovered_items_relocates_stale_row() -> None:
    relocated = _Entity("old/path", "old")
    stale = [relocated]
    flushed = False

    async def flush() -> None:
        nonlocal flushed
        flushed = True

    await sync_discovered_items(
        discovered=[_Discovered("new/path", "new")],
        stale_items=stale,
        existing_paths=set(),
        get_discovered_path=lambda item: item.relative_path,
        find_relocated=lambda stale_items, _item: stale_items[0],
        apply_relocation=lambda entity, item: setattr(entity, "relative_path", item.relative_path),
        stage=lambda _entity: None,
        flush=flush,
        make_unique_name=lambda _item: _async_value("unused"),
        create_entity=lambda _item, _name: _noop(),
    )

    assert relocated.relative_path == "new/path"
    assert stale == []
    assert flushed is True


async def _async_value(value: str) -> str:
    return value


async def _noop() -> None:
    return None
