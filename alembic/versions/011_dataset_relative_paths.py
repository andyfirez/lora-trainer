"""Migrate datasets to relative paths and drop caption_dir."""

from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _default_datasets_root() -> Path:
    return Path("~/lora-trainer/datasets").expanduser().resolve()


def _to_relative(image_dir: str, root: Path) -> str:
    try:
        resolved = Path(image_dir).expanduser().resolve()
    except OSError:
        return image_dir
    try:
        rel = resolved.relative_to(root)
        return "" if rel == Path(".") else rel.as_posix()
    except ValueError:
        return image_dir


def upgrade() -> None:
    bind = op.get_bind()
    root = _default_datasets_root()

    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.add_column(sa.Column("relative_path", sa.String(), nullable=True))

    rows = bind.execute(sa.text("SELECT id, image_dir FROM datasets")).fetchall()
    for row_id, image_dir in rows:
        relative_path = _to_relative(image_dir, root)
        bind.execute(
            sa.text("UPDATE datasets SET relative_path = :relative_path WHERE id = :id"),
            {"relative_path": relative_path, "id": row_id},
        )

    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.alter_column("relative_path", nullable=False)
        batch_op.drop_column("caption_dir")
        batch_op.drop_column("image_dir")


def downgrade() -> None:
    bind = op.get_bind()
    root = _default_datasets_root()

    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.add_column(sa.Column("image_dir", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("caption_dir", sa.String(), nullable=True))

    rows = bind.execute(sa.text("SELECT id, relative_path FROM datasets")).fetchall()
    for row_id, relative_path in rows:
        path = Path(relative_path)
        if path.is_absolute():
            image_dir = str(path)
        else:
            image_dir = str(root / relative_path) if relative_path else str(root)
        bind.execute(
            sa.text("UPDATE datasets SET image_dir = :image_dir WHERE id = :id"),
            {"image_dir": image_dir, "id": row_id},
        )

    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.alter_column("image_dir", nullable=False)
        batch_op.drop_column("relative_path")
