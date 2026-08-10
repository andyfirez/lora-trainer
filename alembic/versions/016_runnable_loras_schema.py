"""Create the loras table (RunnableMixin + LoRA-specific columns).

Schema only — no data touches `jobs` / `trained_loras` yet (see 018).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "loras",
        sa.Column("id", sa.Integer(), nullable=False),
        # RunnableMixin
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="draft"),
        sa.Column("config_yaml", sa.String(), nullable=True),
        sa.Column("queue_position", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("log_path", sa.String(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("running_started_at", sa.DateTime(), nullable=True),
        sa.Column("accumulated_elapsed_seconds", sa.Float(), nullable=False, server_default="0"),
        # LoRA-specific
        sa.Column("relative_path", sa.String(), nullable=False, server_default=""),
        sa.Column("weights_relpath", sa.String(), nullable=False, server_default=""),
        sa.Column("base_model_name", sa.String(), nullable=False, server_default="unknown"),
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
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_loras_name", "loras", ["name"], unique=False)
    op.create_index("ix_loras_status", "loras", ["status"], unique=False)
    op.create_index("ix_loras_queue_position", "loras", ["queue_position"], unique=False)
    op.create_index("ix_loras_base_model_name", "loras", ["base_model_name"], unique=False)


def downgrade() -> None:
    op.drop_table("loras")
