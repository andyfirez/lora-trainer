"""Drop mid-training sampling progress columns from jobs."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("sampling_total")
        batch_op.drop_column("sampling_step")
        batch_op.drop_column("sampling_status")


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("sampling_status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("sampling_step", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("sampling_total", sa.Integer(), nullable=True))
