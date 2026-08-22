"""Create prompt builder tables."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prompt_sections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prompt_sections_name", "prompt_sections", ["name"], unique=False)
    op.create_index("ix_prompt_sections_sort_order", "prompt_sections", ["sort_order"], unique=False)

    op.create_table(
        "prompt_section_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("section_id", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["section_id"], ["prompt_sections.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prompt_section_values_section_id",
        "prompt_section_values",
        ["section_id"],
        unique=False,
    )
    op.create_index(
        "ix_prompt_section_values_sort_order",
        "prompt_section_values",
        ["sort_order"],
        unique=False,
    )

    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("polarity", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prompt_templates_name", "prompt_templates", ["name"], unique=False)
    op.create_index("ix_prompt_templates_polarity", "prompt_templates", ["polarity"], unique=False)

    op.create_table(
        "prompt_template_sections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("section_id", sa.Integer(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("selection_mode", sa.String(), nullable=False, server_default="fixed"),
        sa.Column("selected_value_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["section_id"], ["prompt_sections.id"]),
        sa.ForeignKeyConstraint(["selected_value_id"], ["prompt_section_values.id"]),
        sa.ForeignKeyConstraint(["template_id"], ["prompt_templates.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prompt_template_sections_template_id",
        "prompt_template_sections",
        ["template_id"],
        unique=False,
    )
    op.create_index(
        "ix_prompt_template_sections_section_id",
        "prompt_template_sections",
        ["section_id"],
        unique=False,
    )
    op.create_index(
        "ix_prompt_template_sections_sort_order",
        "prompt_template_sections",
        ["sort_order"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("prompt_template_sections")
    op.drop_table("prompt_templates")
    op.drop_table("prompt_section_values")
    op.drop_table("prompt_sections")
