"""Normalize absolute dataset relative_path values to paths relative to datasets_root."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from src.db.migrations.dataset_paths import datasets_root_from_config, normalize_dataset_relative_path

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
    root = datasets_root_from_config()

    rows = connection.execute(sa.text("SELECT id, relative_path FROM datasets")).fetchall()
    for row_id, relative_path in rows:
        normalized = normalize_dataset_relative_path(relative_path, root)
        if normalized == relative_path:
            continue
        connection.execute(
            sa.text("UPDATE datasets SET relative_path = :relative_path WHERE id = :id"),
            {"relative_path": normalized, "id": row_id},
        )

    _dedupe_datasets(connection)


def downgrade() -> None:
    pass
