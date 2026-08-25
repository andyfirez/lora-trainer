"""Add samplings.kind to distinguish experiments from playground runs."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("samplings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("kind", sa.String(), nullable=False, server_default="playground"),
        )
        batch_op.create_index("ix_samplings_kind", ["kind"], unique=False)

    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            UPDATE samplings
            SET kind = 'experiment'
            WHERE json_extract(config, '$.compose_grids') IN (1, 'true', 'True')
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("samplings", schema=None) as batch_op:
        batch_op.drop_index("ix_samplings_kind")
        batch_op.drop_column("kind")
