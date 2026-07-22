"""Add trained_loras catalog and remove config versioning."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _runtime_lora_name(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml) or {}
    return str(data.get("lora_name", "lora"))


def _base_model_name(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml) or {}
    return str(data.get("base_model_name", "unknown"))


def _output_format(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml) or {}
    return str(data.get("output_format", "safetensors"))


def _resolve_paths(config_yaml: str, output_path: str | None) -> tuple[str, str] | None:
    data = yaml.safe_load(config_yaml) or {}
    lora_name = str(data.get("lora_name", "lora"))
    output_dir = data.get("output_dir")
    if output_path:
        work_dir = Path(output_path)
    elif isinstance(output_dir, str) and output_dir:
        work_dir = Path(output_dir) / lora_name
    else:
        return None
    ext = _output_format(config_yaml)
    if ext == "OutputFormat.SAFETENSORS":
        ext = "safetensors"
    weights_path = work_dir / f"{lora_name}.{ext}"
    if not weights_path.is_file():
        return None
    return str(weights_path), str(work_dir)


def upgrade() -> None:
    op.create_table(
        "trained_loras",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("config_id", sa.Integer(), nullable=True),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("base_model_name", sa.String(), nullable=False),
        sa.Column("weights_path", sa.String(), nullable=False),
        sa.Column("work_dir", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    with op.batch_alter_table("trained_loras", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_trained_loras_base_model_name"), ["base_model_name"], unique=False)
        batch_op.create_index(batch_op.f("ix_trained_loras_config_id"), ["config_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_trained_loras_job_id"), ["job_id"], unique=True)
        batch_op.create_index(batch_op.f("ix_trained_loras_name"), ["name"], unique=False)

    connection = op.get_bind()
    now = datetime.now(timezone.utc).isoformat()
    rows = connection.execute(
        sa.text(
            "SELECT id, config_id, config_yaml, output_path FROM jobs "
            "WHERE job_type = 'training' AND status = 'completed'"
        )
    ).fetchall()
    for job_id, config_id, config_yaml, output_path in rows:
        if not config_yaml:
            continue
        resolved = _resolve_paths(config_yaml, output_path)
        if resolved is None:
            continue
        weights_path, work_dir = resolved
        connection.execute(
            sa.text(
                "INSERT INTO trained_loras "
                "(name, job_id, config_id, config_yaml, base_model_name, weights_path, work_dir, created_at, updated_at) "
                "VALUES (:name, :job_id, :config_id, :config_yaml, :base_model_name, :weights_path, :work_dir, :created_at, :updated_at)"
            ),
            {
                "name": _runtime_lora_name(config_yaml),
                "job_id": job_id,
                "config_id": config_id,
                "config_yaml": config_yaml,
                "base_model_name": _base_model_name(config_yaml),
                "weights_path": weights_path,
                "work_dir": work_dir,
                "created_at": now,
                "updated_at": now,
            },
        )

    with op.batch_alter_table("job_config_versions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_job_config_versions_config_id"))
    op.drop_table("job_config_versions")

    with op.batch_alter_table("job_configs", schema=None) as batch_op:
        batch_op.drop_column("active_version")

    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("config_version")


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("config_version", sa.Integer(), nullable=True))

    with op.batch_alter_table("job_configs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("active_version", sa.Integer(), nullable=True))

    op.create_table(
        "job_config_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("config_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("config_id", "version", name="uq_job_config_versions_config_version"),
    )
    with op.batch_alter_table("job_config_versions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_job_config_versions_config_id"), ["config_id"], unique=False)

    with op.batch_alter_table("trained_loras", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_trained_loras_name"))
        batch_op.drop_index(batch_op.f("ix_trained_loras_job_id"))
        batch_op.drop_index(batch_op.f("ix_trained_loras_config_id"))
        batch_op.drop_index(batch_op.f("ix_trained_loras_base_model_name"))
    op.drop_table("trained_loras")
