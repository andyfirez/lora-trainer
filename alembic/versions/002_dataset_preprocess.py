"""Add dataset preprocessing fields and crop metadata table."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.add_column(sa.Column("target_resolution", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("preprocess_ready", sa.Boolean(), nullable=False, server_default=sa.false())
        )

    op.create_table(
        "dataset_image_crops",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("crop_center_x", sa.Float(), nullable=False),
        sa.Column("crop_center_y", sa.Float(), nullable=False),
        sa.Column("source_mtime", sa.Float(), nullable=False),
        sa.Column("baked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "filename", name="uq_dataset_image_crops_dataset_filename"),
    )
    with op.batch_alter_table("dataset_image_crops", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_dataset_image_crops_dataset_id"),
            ["dataset_id"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("dataset_image_crops")
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.drop_column("preprocess_ready")
        batch_op.drop_column("target_resolution")
