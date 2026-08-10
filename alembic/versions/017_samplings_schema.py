"""Create the samplings table (RunnableMixin + sampling-specific columns).

Schema only — no data touches `jobs` yet (see 018).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "samplings",
        sa.Column("id", sa.Integer(), nullable=False),
        # RunnableMixin
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="draft"),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("queue_position", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("log_path", sa.String(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("running_started_at", sa.DateTime(), nullable=True),
        sa.Column("accumulated_elapsed_seconds", sa.Float(), nullable=False, server_default="0"),
        # Sampling-specific
        sa.Column("lora_paths_yaml", sa.String(), nullable=True),
        sa.Column("progress_step", sa.Integer(), nullable=True),
        sa.Column("progress_total", sa.Integer(), nullable=True),
        sa.Column("progress_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_samplings_name", "samplings", ["name"], unique=False)
    op.create_index("ix_samplings_status", "samplings", ["status"], unique=False)
    op.create_index("ix_samplings_queue_position", "samplings", ["queue_position"], unique=False)


def downgrade() -> None:
    op.drop_table("samplings")
