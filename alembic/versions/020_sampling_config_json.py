"""Store sampling configs as JSON instead of YAML."""

from __future__ import annotations

import json
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _load_yaml(value: str | None, fallback):
    try:
        data = yaml.safe_load(value or "") or fallback
    except Exception:
        return fallback
    return data if isinstance(data, type(fallback)) else fallback


def upgrade() -> None:
    with op.batch_alter_table("samplings", schema=None) as batch_op:
        batch_op.add_column(sa.Column("config", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("lora_paths", sa.JSON(), nullable=True))

    connection = op.get_bind()
    rows = connection.execute(sa.text("SELECT id, config_yaml, lora_paths_yaml FROM samplings")).fetchall()
    for row_id, config_yaml, lora_paths_yaml in rows:
        config = _load_yaml(config_yaml, {})
        lora_paths = _load_yaml(lora_paths_yaml, [])
        connection.execute(
            sa.text("UPDATE samplings SET config = :config, lora_paths = :lora_paths WHERE id = :id"),
            {"config": json.dumps(config), "lora_paths": json.dumps(lora_paths), "id": row_id},
        )

    with op.batch_alter_table("samplings", schema=None) as batch_op:
        batch_op.alter_column("config", existing_type=sa.JSON(), nullable=False)
        batch_op.drop_column("config_yaml")
        batch_op.drop_column("lora_paths_yaml")


def downgrade() -> None:
    with op.batch_alter_table("samplings", schema=None) as batch_op:
        batch_op.add_column(sa.Column("config_yaml", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("lora_paths_yaml", sa.String(), nullable=True))

    connection = op.get_bind()
    rows = connection.execute(sa.text("SELECT id, config, lora_paths FROM samplings")).fetchall()
    for row_id, config, lora_paths in rows:
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                config = {}
        if not isinstance(config, dict):
            config = {}
        if isinstance(lora_paths, str):
            try:
                lora_paths = json.loads(lora_paths)
            except Exception:
                lora_paths = []
        if not isinstance(lora_paths, list):
            lora_paths = []
        connection.execute(
            sa.text(
                "UPDATE samplings SET config_yaml = :config_yaml, lora_paths_yaml = :lora_paths_yaml WHERE id = :id"
            ),
            {
                "config_yaml": yaml.dump(config, allow_unicode=True, sort_keys=False),
                "lora_paths_yaml": yaml.dump(lora_paths, allow_unicode=True, sort_keys=False),
                "id": row_id,
            },
        )

    with op.batch_alter_table("samplings", schema=None) as batch_op:
        batch_op.alter_column("config_yaml", existing_type=sa.String(), nullable=False)
        batch_op.drop_column("config")
        batch_op.drop_column("lora_paths")
