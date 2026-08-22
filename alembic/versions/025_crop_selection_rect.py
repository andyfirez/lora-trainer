"""Add crop selection rectangle fields for zoom crop."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.add_column(sa.Column("selection_x", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("selection_y", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("selection_width", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("selection_height", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.drop_column("selection_height")
        batch_op.drop_column("selection_width")
        batch_op.drop_column("selection_y")
        batch_op.drop_column("selection_x")
