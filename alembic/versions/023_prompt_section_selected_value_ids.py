"""Add selected_value_ids JSON column to prompt_template_sections."""

from __future__ import annotations

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("prompt_template_sections", schema=None) as batch_op:
        batch_op.add_column(sa.Column("selected_value_ids", sa.JSON(), nullable=True))

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT pts.id, pts.selection_mode, pts.selected_value_id, pts.section_id "
            "FROM prompt_template_sections pts"
        )
    ).fetchall()
    for link_id, selection_mode, selected_value_id, section_id in rows:
        if selection_mode == "random":
            value_rows = connection.execute(
                sa.text(
                    "SELECT id FROM prompt_section_values "
                    "WHERE section_id = :section_id ORDER BY sort_order, id"
                ),
                {"section_id": section_id},
            ).fetchall()
            ids = [row[0] for row in value_rows]
        elif selected_value_id is not None:
            ids = [selected_value_id]
        else:
            ids = []
        connection.execute(
            sa.text("UPDATE prompt_template_sections SET selected_value_ids = :ids WHERE id = :id"),
            {"ids": json.dumps(ids), "id": link_id},
        )


def downgrade() -> None:
    with op.batch_alter_table("prompt_template_sections", schema=None) as batch_op:
        batch_op.drop_column("selected_value_ids")
