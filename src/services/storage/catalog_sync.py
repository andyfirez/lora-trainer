"""Generic sync of on-disk catalog discoveries into the database."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

D = TypeVar("D")
E = TypeVar("E")


async def sync_discovered_items(
    *,
    discovered: Sequence[D],
    stale_items: list[E],
    existing_paths: set[str],
    get_discovered_path: Callable[[D], str],
    find_relocated: Callable[[list[E], D], E | None],
    apply_relocation: Callable[[E, D], None],
    stage: Callable[[E], None],
    flush: Callable[[], Awaitable[None]],
    on_known_path: Callable[[D], bool] | None = None,
    make_unique_name: Callable[[D], Awaitable[str]],
    create_entity: Callable[[D, str], Awaitable[None]],
) -> None:
    """Reconcile discovered on-disk paths with catalog rows.

    Shared algorithm used by LoRA and dataset discovery:
    skip known paths (optional update hook) -> relocation -> unique name -> create.
    """
    changed = False

    for item in discovered:
        path = get_discovered_path(item)
        if path in existing_paths:
            if on_known_path is not None and on_known_path(item):
                changed = True
            continue

        relocated = find_relocated(stale_items, item)
        if relocated is not None:
            apply_relocation(relocated, item)
            stale_items.remove(relocated)
            existing_paths.add(path)
            stage(relocated)
            changed = True
            continue

        unique_name = await make_unique_name(item)
        await create_entity(item, unique_name)
        existing_paths.add(path)

    if changed:
        await flush()
