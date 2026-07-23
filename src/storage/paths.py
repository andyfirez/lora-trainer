"""Resolve managed storage paths relative to configured roots."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from src.settings.app_settings import settings


class StorageKind(StrEnum):
    DATASETS = "datasets"
    BASE_MODELS = "base_models"
    LORA = "lora"


def _expand(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


class StoragePaths:
    """Central path resolver for managed storage roots."""

    @staticmethod
    def datasets_root() -> Path:
        return _expand(settings.storage.datasets_root)

    @staticmethod
    def base_models_root() -> Path:
        return _expand(settings.storage.base_models_root)

    @staticmethod
    def lora_root() -> Path:
        return _expand(settings.storage.lora_root)

    @classmethod
    def root_for(cls, kind: StorageKind) -> Path:
        if kind == StorageKind.DATASETS:
            return cls.datasets_root()
        if kind == StorageKind.BASE_MODELS:
            return cls.base_models_root()
        return cls.lora_root()

    @classmethod
    def resolve(cls, kind: StorageKind, relative_path: str) -> Path:
        rel = relative_path.strip().strip("/\\")
        root = cls.root_for(kind)
        if not rel:
            return root
        candidate = (root / rel).resolve()
        if not cls._is_under_root(candidate, root):
            raise ValueError(f"Path escapes managed root: {relative_path}")
        return candidate

    @classmethod
    def resolve_dataset_path(cls, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path.expanduser().resolve()
        return cls.resolve(StorageKind.DATASETS, relative_path)

    @classmethod
    def is_managed_relative_path(cls, kind: StorageKind, relative_path: str) -> bool:
        if Path(relative_path).is_absolute():
            return cls.to_relative(kind, relative_path) is not None
        return True

    @classmethod
    def resolve_training_work_dir(cls, output_dir_relative: str, lora_name: str) -> Path:
        return cls.resolve_lora_path(output_dir_relative) / lora_name

    @classmethod
    def resolve_base_model(cls, relative_path: str) -> Path:
        return cls.resolve(StorageKind.BASE_MODELS, relative_path)

    @classmethod
    def base_model_exists(cls, relative_path: str) -> bool:
        try:
            path = cls.resolve_base_model(relative_path)
        except ValueError:
            return False
        return path.is_file() or path.is_dir()

    @classmethod
    def resolve_base_model_path(cls, relative_path: str) -> Path:
        return cls.resolve(StorageKind.BASE_MODELS, relative_path)

    @classmethod
    def resolve_lora_path(cls, relative_path: str) -> Path:
        return cls.resolve(StorageKind.LORA, relative_path)

    @classmethod
    def to_relative(cls, kind: StorageKind, absolute_path: str | Path) -> str | None:
        root = cls.root_for(kind)
        try:
            resolved = Path(absolute_path).expanduser().resolve()
        except OSError:
            return None
        if not cls._is_under_root(resolved, root):
            return None
        rel = resolved.relative_to(root)
        return "" if rel == Path(".") else rel.as_posix()

    @classmethod
    def path_exists(cls, kind: StorageKind, relative_path: str) -> bool:
        try:
            return cls.resolve(kind, relative_path).exists()
        except ValueError:
            return False

    @classmethod
    def ensure_root(cls, kind: StorageKind) -> Path:
        root = cls.root_for(kind)
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _is_under_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_relative_path(cls, kind: StorageKind, relative_path: str) -> str:
        rel = relative_path.strip().strip("/\\")
        if ".." in Path(rel).parts:
            raise ValueError("Relative path must not contain parent segments")
        cls.resolve(kind, rel)
        return rel
