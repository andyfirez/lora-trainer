"""Browse managed storage directories."""

from dataclasses import dataclass
from pathlib import Path

from src.storage.paths import StorageKind, StoragePaths


@dataclass(frozen=True)
class StorageEntry:
    name: str
    relative_path: str
    is_dir: bool


class StorageBrowseService:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def list_entries(self, kind: StorageKind, relative_path: str = "") -> list[StorageEntry]:
        root = StoragePaths.root_for(kind)
        target = StoragePaths.resolve(kind, relative_path) if relative_path else root
        if not target.is_dir():
            return []

        entries: list[StorageEntry] = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            rel = StoragePaths.to_relative(kind, child)
            if rel is None:
                continue
            entries.append(StorageEntry(name=child.name, relative_path=rel, is_dir=child.is_dir()))
        return entries

    def list_model_entries(self, relative_path: str = "") -> list[StorageEntry]:
        entries = self.list_entries(StorageKind.BASE_MODELS, relative_path)
        result: list[StorageEntry] = []
        for entry in entries:
            if entry.is_dir:
                result.append(entry)
                continue
            if entry.name.endswith((".safetensors", ".ckpt")):
                result.append(entry)
        return result

    def discover_dataset_folders(self) -> list[str]:
        root = StoragePaths.datasets_root()
        if not root.is_dir():
            return []

        discovered: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            if self._folder_has_images(path):
                rel = StoragePaths.to_relative(StorageKind.DATASETS, path)
                if rel is not None:
                    discovered.append(rel)
        return discovered

    @classmethod
    def _folder_has_images(cls, folder: Path) -> bool:
        return any(
            child.is_file() and child.suffix.lower() in cls.IMAGE_EXTENSIONS
            for child in folder.iterdir()
            if not child.name.startswith(".")
        )
