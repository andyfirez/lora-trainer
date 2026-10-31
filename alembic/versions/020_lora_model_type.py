"""Add loras.model_type (base model family: sdxl, flux1, flux2_klein).

Existing rows are SDXL LoRAs; their stored YAML needs no rewrite because
``TrainConfig.model_type`` defaults to ``sdxl``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("loras", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("model_type", sa.String(), nullable=False, server_default="sdxl")
        )
        batch_op.create_index("ix_loras_model_type", ["model_type"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("loras", schema=None) as batch_op:
        batch_op.drop_index("ix_loras_model_type")
        batch_op.drop_column("model_type")
