"""Add is_workspace flag to prompt_templates."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("prompt_templates", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("is_workspace", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    op.create_index("ix_prompt_templates_is_workspace", "prompt_templates", ["is_workspace"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_prompt_templates_is_workspace", table_name="prompt_templates")
    with op.batch_alter_table("prompt_templates", schema=None) as batch_op:
        batch_op.drop_column("is_workspace")
