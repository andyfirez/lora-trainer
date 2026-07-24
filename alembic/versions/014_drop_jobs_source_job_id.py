"""Drop jobs.source_job_id — sampling is no longer linked to training jobs."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_jobs_source_job_id"))
        batch_op.drop_column("source_job_id")


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_job_id", sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f("ix_jobs_source_job_id"), ["source_job_id"], unique=False)
