"""Drop legacy queue_entries, trained_loras, jobs, job_configs tables.

All data was copied into loras/samplings by 018. Downgrade is best-effort
(recreates empty table shells only — merged rows cannot be split back).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("queue_entries")
    op.drop_table("trained_loras")
    op.drop_table("jobs")
    op.drop_table("job_configs")


def downgrade() -> None:
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
    op.create_index("ix_job_configs_name", "job_configs", ["name"], unique=False)
    op.create_index("ix_job_configs_config_type", "job_configs", ["config_type"], unique=False)

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
        sa.Column("last_checkpoint_path", sa.String(), nullable=True),
        sa.Column("last_checkpoint_epoch", sa.Integer(), nullable=True),
        sa.Column("last_checkpoint_step", sa.Integer(), nullable=True),
        sa.Column("resume_checkpoint_path", sa.String(), nullable=True),
        sa.Column("resume_from_epoch", sa.Integer(), nullable=True),
        sa.Column("resume_from_step", sa.Integer(), nullable=True),
        sa.Column("save_checkpoint_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("lora_paths_yaml", sa.String(), nullable=True),
        sa.Column("progress_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("running_started_at", sa.DateTime(), nullable=True),
        sa.Column("accumulated_elapsed_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"], unique=False)
    op.create_index("ix_jobs_name", "jobs", ["name"], unique=False)
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_config_id", "jobs", ["config_id"], unique=False)

    op.create_table(
        "trained_loras",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("relative_path", sa.String(), nullable=False),
        sa.Column("weights_relpath", sa.String(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("config_id", sa.Integer(), nullable=True),
        sa.Column("config_yaml", sa.String(), nullable=True),
        sa.Column("base_model_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_trained_loras_name", "trained_loras", ["name"], unique=False)
    op.create_index("ix_trained_loras_base_model_name", "trained_loras", ["base_model_name"], unique=False)
    op.create_index("ix_trained_loras_config_id", "trained_loras", ["config_id"], unique=False)
    op.create_index("ix_trained_loras_job_id", "trained_loras", ["job_id"], unique=True)

    op.create_table(
        "queue_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("added_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_queue_entries_job_id", "queue_entries", ["job_id"], unique=False)
    op.create_index("ix_queue_entries_position", "queue_entries", ["position"], unique=False)
