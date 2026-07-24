"""Discover LoRA work directories under managed lora_root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.services.loras.weights import pick_weights_file, work_dir_has_weights
from src.storage.paths import StorageKind, StoragePaths


@dataclass(frozen=True)
class DiscoveredLora:
    relative_path: str
    weights_relpath: str
    name: str


def _to_relative(root: Path, path: Path) -> str | None:
    try:
        rel = path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except (ValueError, OSError):
        return None
    return "" if rel == Path(".") else rel.as_posix()


class LoraDiscoveryService:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    def _root_path(self) -> Path:
        return self._root if self._root is not None else StoragePaths.lora_root()

    def discover_lora_work_dirs(self) -> list[DiscoveredLora]:
        root = self._root_path()
        if not root.is_dir():
            return []

        discovered: list[DiscoveredLora] = []
        seen_paths: set[str] = set()
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            if not work_dir_has_weights(path):
                continue
            rel = _to_relative(root, path)
            if rel is None or rel in seen_paths:
                continue
            weights = pick_weights_file(path)
            if weights is None:
                continue
            weights_rel = _to_relative(root, weights)
            if weights_rel is None:
                continue
            seen_paths.add(rel)
            discovered.append(
                DiscoveredLora(
                    relative_path=rel,
                    weights_relpath=weights_rel,
                    name=Path(rel).name or rel.replace("/", "-").replace("\\", "-") or "lora",
                )
            )
        return discovered
