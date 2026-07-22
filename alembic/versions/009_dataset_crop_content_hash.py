"""Add content_hash to dataset_image_crops for rename detection during reconcile."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.add_column(sa.Column("content_hash", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.drop_column("content_hash")
