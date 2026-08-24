"""Default enable_bucket to true for new datasets."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.alter_column(
            "enable_bucket",
            existing_type=sa.Boolean(),
            server_default=sa.true(),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.alter_column(
            "enable_bucket",
            existing_type=sa.Boolean(),
            server_default=sa.false(),
            existing_nullable=False,
        )
