"""Add include_empty flag to prompt_template_sections."""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("prompt_template_sections", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("include_empty", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    with op.batch_alter_table("prompt_template_sections", schema=None) as batch_op:
        batch_op.drop_column("include_empty")
