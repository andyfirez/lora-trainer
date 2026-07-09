"""Tests for frontend parameter metadata coverage."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"


def test_parameter_metadata_validation_script() -> None:
    result = subprocess.run(
        ["node", "scripts/validate-parameter-metadata.mjs"],
        cwd=FRONTEND_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
