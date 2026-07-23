"""Migrate legacy training config YAML (learning_rate, inline sampling fields)."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from src.db.migrations.training_yaml import migrate_training_yaml

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TRAINING_YAML_TABLES: tuple[tuple[str, str | None], ...] = (
    ("job_configs", "config_type = 'training'"),
    ("jobs", "job_type = 'training'"),
    ("trained_loras", None),
)


def _migrate_table(connection: sa.Connection, table: str, where_clause: str | None) -> None:
    query = f"SELECT id, config_yaml FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    rows = connection.execute(sa.text(query)).fetchall()
    for row_id, config_yaml in rows:
        if not config_yaml:
            continue
        migrated = migrate_training_yaml(config_yaml)
        if migrated is None:
            continue
        connection.execute(
            sa.text(f"UPDATE {table} SET config_yaml = :config_yaml WHERE id = :id"),
            {"config_yaml": migrated, "id": row_id},
        )


def upgrade() -> None:
    connection = op.get_bind()
    for table, where_clause in _TRAINING_YAML_TABLES:
        _migrate_table(connection, table, where_clause)


def downgrade() -> None:
    pass
