"""Normalize legacy config YAML stored in job_configs, jobs, and trained_loras."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from src.db.migrations.sampling_yaml import migrate_sampling_yaml
from src.db.migrations.training_yaml import migrate_training_yaml

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONFIG_YAML_TABLES: tuple[tuple[str, str | None, str], ...] = (
    ("job_configs", "config_type = 'sampling'", "sampling"),
    ("jobs", "job_type = 'sampling'", "sampling"),
    ("job_configs", "config_type = 'training'", "training"),
    ("jobs", "job_type = 'training'", "training"),
    ("trained_loras", None, "training"),
)


def _migrate_table(connection: sa.Connection, table: str, where_clause: str | None, kind: str) -> None:
    query = f"SELECT id, config_yaml FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    rows = connection.execute(sa.text(query)).fetchall()
    migrate = migrate_sampling_yaml if kind == "sampling" else migrate_training_yaml
    for row_id, config_yaml in rows:
        if not config_yaml:
            continue
        migrated = migrate(config_yaml)
        if migrated is None:
            continue
        connection.execute(
            sa.text(f"UPDATE {table} SET config_yaml = :config_yaml WHERE id = :id"),
            {"config_yaml": migrated, "id": row_id},
        )


def upgrade() -> None:
    connection = op.get_bind()
    for table, where_clause, kind in _CONFIG_YAML_TABLES:
        _migrate_table(connection, table, where_clause, kind)


def downgrade() -> None:
    pass
