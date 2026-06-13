"""Native filesystem dialogs via tkinter."""

import asyncio
from enum import StrEnum
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from src.services.files.exceptions import PickCancelledError

MODEL_FILE_TYPES: list[tuple[str, str]] = [
    ("Model files", "*.safetensors *.ckpt *.bin *.pt *.pth"),
    ("Safetensors", "*.safetensors"),
    ("All files", "*.*"),
]


class PickKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    MODEL = "model"


class FilesService:
    async def pick_path(
        self,
        kind: PickKind,
        *,
        title: str,
        initial_path: str | None = None,
    ) -> str:
        initial_dir = self._resolve_initial_dir(initial_path)
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(
            None,
            _pick_path_sync,
            kind,
            title,
            initial_dir,
        )
        if not path:
            raise PickCancelledError()
        return path

    def _resolve_initial_dir(self, initial_path: str | None) -> str | None:
        if not initial_path:
            return None
        path = Path(initial_path).expanduser()
        if path.is_file():
            return str(path.parent)
        if path.is_dir():
            return str(path)
        parent = path.parent
        return str(parent) if parent.exists() else None


def _pick_path_sync(kind: PickKind, title: str, initial_dir: str | None) -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == PickKind.DIRECTORY:
            return filedialog.askdirectory(title=title, initialdir=initial_dir, mustexist=True) or None
        filetypes = MODEL_FILE_TYPES if kind == PickKind.MODEL else [("All files", "*.*")]
        return filedialog.askopenfilename(title=title, initialdir=initial_dir, filetypes=filetypes) or None
    finally:
        root.destroy()
