"""Initial schema: datasets, job_configs, jobs, queue_entries."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("image_dir", sa.String(), nullable=False),
        sa.Column("caption_dir", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_datasets_name"), ["name"], unique=True)

    op.create_table(
        "job_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("config_type", sa.String(), nullable=False),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("job_configs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_job_configs_name"), ["name"], unique=False)
        batch_op.create_index(batch_op.f("ix_job_configs_config_type"), ["config_type"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("config_id", sa.Integer(), nullable=True),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("log_path", sa.String(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("progress_step", sa.Integer(), nullable=True),
        sa.Column("progress_total", sa.Integer(), nullable=True),
        sa.Column("progress_loss", sa.Float(), nullable=True),
        sa.Column("progress_avr_loss", sa.Float(), nullable=True),
        sa.Column("progress_epoch", sa.Integer(), nullable=True),
        sa.Column("progress_epoch_total", sa.Integer(), nullable=True),
        sa.Column("cache_progress_step", sa.Integer(), nullable=True),
        sa.Column("cache_progress_total", sa.Integer(), nullable=True),
        sa.Column("sampling_status", sa.String(), nullable=True),
        sa.Column("sampling_step", sa.Integer(), nullable=True),
        sa.Column("sampling_total", sa.Integer(), nullable=True),
        sa.Column("last_checkpoint_path", sa.String(), nullable=True),
        sa.Column("last_checkpoint_epoch", sa.Integer(), nullable=True),
        sa.Column("last_checkpoint_step", sa.Integer(), nullable=True),
        sa.Column("resume_checkpoint_path", sa.String(), nullable=True),
        sa.Column("resume_from_epoch", sa.Integer(), nullable=True),
        sa.Column("resume_from_step", sa.Integer(), nullable=True),
        sa.Column("save_checkpoint_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("lora_paths_yaml", sa.String(), nullable=True),
        sa.Column("source_job_id", sa.Integer(), nullable=True),
        sa.Column("progress_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.ForeignKeyConstraint(["source_job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_jobs_job_type"), ["job_type"], unique=False)
        batch_op.create_index(batch_op.f("ix_jobs_name"), ["name"], unique=False)
        batch_op.create_index(batch_op.f("ix_jobs_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_jobs_config_id"), ["config_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_jobs_source_job_id"), ["source_job_id"], unique=False)

    op.create_table(
        "queue_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("added_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("queue_entries", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_queue_entries_job_id"), ["job_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_queue_entries_position"), ["position"], unique=False)


def downgrade() -> None:
    op.drop_table("queue_entries")
    op.drop_table("jobs")
    op.drop_table("job_configs")
    op.drop_table("datasets")
