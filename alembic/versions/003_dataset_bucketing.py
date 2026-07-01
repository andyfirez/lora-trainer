"""Add aspect-ratio bucketing fields to datasets and crop metadata."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("enable_bucket", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(
            sa.Column("bucket_reso_steps", sa.Integer(), nullable=False, server_default="64")
        )
        batch_op.add_column(
            sa.Column("min_bucket_reso", sa.Integer(), nullable=False, server_default="512")
        )
        batch_op.add_column(
            sa.Column("max_bucket_reso", sa.Integer(), nullable=False, server_default="2048")
        )
        batch_op.add_column(
            sa.Column("bucket_no_upscale", sa.Boolean(), nullable=False, server_default=sa.true())
        )

    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.add_column(sa.Column("bucket_width", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("bucket_height", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("scale_to_width", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("scale_to_height", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("crop_x", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("crop_y", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.drop_column("crop_y")
        batch_op.drop_column("crop_x")
        batch_op.drop_column("scale_to_height")
        batch_op.drop_column("scale_to_width")
        batch_op.drop_column("bucket_height")
        batch_op.drop_column("bucket_width")

    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.drop_column("bucket_no_upscale")
        batch_op.drop_column("max_bucket_reso")
        batch_op.drop_column("min_bucket_reso")
        batch_op.drop_column("bucket_reso_steps")
        batch_op.drop_column("enable_bucket")
