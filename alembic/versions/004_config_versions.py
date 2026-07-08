"""Add training config version history and job config_version."""

import re
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LORA_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def _strip_lora_version_suffix(name: str) -> str:
    return _LORA_VERSION_SUFFIX_RE.sub("", name)


def _apply_lora_version_to_yaml(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml) or {}
    lora_name = str(data.get("lora_name", "lora"))
    base_name = _strip_lora_version_suffix(lora_name)
    data["lora_name"] = f"{base_name}_v1"
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def upgrade() -> None:
    op.create_table(
        "job_config_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("config_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("config_id", "version", name="uq_job_config_versions_config_version"),
    )
    with op.batch_alter_table("job_config_versions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_job_config_versions_config_id"), ["config_id"], unique=False)

    with op.batch_alter_table("job_configs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("active_version", sa.Integer(), nullable=True))

    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("config_version", sa.Integer(), nullable=True))

    connection = op.get_bind()
    training_configs = connection.execute(
        sa.text("SELECT id, config_yaml FROM job_configs WHERE config_type = 'training'")
    ).fetchall()
    now = datetime.now(timezone.utc)
    for config_id, config_yaml in training_configs:
        versioned_yaml = _apply_lora_version_to_yaml(config_yaml)
        connection.execute(
            sa.text(
                "INSERT INTO job_config_versions (config_id, version, config_yaml, created_at) "
                "VALUES (:config_id, 1, :config_yaml, :created_at)"
            ),
            {"config_id": config_id, "config_yaml": versioned_yaml, "created_at": now},
        )
        connection.execute(
            sa.text(
                "UPDATE job_configs SET config_yaml = :config_yaml, active_version = 1 WHERE id = :config_id"
            ),
            {"config_id": config_id, "config_yaml": versioned_yaml},
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("config_version")

    with op.batch_alter_table("job_configs", schema=None) as batch_op:
        batch_op.drop_column("active_version")

    with op.batch_alter_table("job_config_versions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_job_config_versions_config_id"))

    op.drop_table("job_config_versions")
