"""Normalize absolute dataset relative_path values to paths relative to datasets_root."""

import os
import tomllib
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _datasets_root_from_config(config_path: Path | None = None) -> Path:
    path = config_path or Path(os.environ.get("APP_CONFIG_FILE", "config.toml"))
    if path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        storage = data.get("storage")
        if isinstance(storage, dict) and storage.get("datasets_root"):
            return Path(str(storage["datasets_root"])).expanduser().resolve()
    return Path("~/lora-trainer/datasets").expanduser().resolve()


def _normalize_dataset_relative_path(relative_path: str, root: Path) -> str:
    path = Path(relative_path)
    if not path.is_absolute():
        return relative_path.strip().strip("/\\").replace("\\", "/")

    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return relative_path

    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return relative_path

    return "" if rel == Path(".") else rel.as_posix()


def _dedupe_datasets(connection: sa.Connection) -> None:
    rows = connection.execute(
        sa.text("SELECT id, relative_path FROM datasets ORDER BY id")
    ).fetchall()
    seen: dict[str, int] = {}
    for row_id, relative_path in rows:
        canonical = relative_path.strip().strip("/\\").replace("\\", "/")
        keeper_id = seen.get(canonical)
        if keeper_id is None:
            seen[canonical] = row_id
            continue

        connection.execute(
            sa.text(
                "UPDATE dataset_image_crops SET dataset_id = :keeper_id "
                "WHERE dataset_id = :duplicate_id"
            ),
            {"keeper_id": keeper_id, "duplicate_id": row_id},
        )
        connection.execute(
            sa.text("DELETE FROM datasets WHERE id = :duplicate_id"),
            {"duplicate_id": row_id},
        )


def upgrade() -> None:
    connection = op.get_bind()
    root = _datasets_root_from_config()

    rows = connection.execute(sa.text("SELECT id, relative_path FROM datasets")).fetchall()
    for row_id, relative_path in rows:
        normalized = _normalize_dataset_relative_path(relative_path, root)
        if normalized == relative_path:
            continue
        connection.execute(
            sa.text("UPDATE datasets SET relative_path = :relative_path WHERE id = :id"),
            {"relative_path": normalized, "id": row_id},
        )

    _dedupe_datasets(connection)


def downgrade() -> None:
    pass
