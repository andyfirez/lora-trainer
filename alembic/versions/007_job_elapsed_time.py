"""Add elapsed time tracking fields to jobs."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("running_started_at", sa.DateTime(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "accumulated_elapsed_seconds",
                sa.Float(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("accumulated_elapsed_seconds")
        batch_op.drop_column("running_started_at")
