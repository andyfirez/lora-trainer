"""LoRA relative paths, discovery backfill, and schema cleanup."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

from src.db.migrations.lora_paths import lora_root_from_config, normalize_lora_relative_path, to_lora_relative
from src.services.loras.discovery import LoraDiscoveryService
from src.services.loras.weights import pick_weights_file

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _base_model_from_yaml(config_yaml: str | None) -> str:
    if not config_yaml:
        return "unknown"
    data = yaml.safe_load(config_yaml) or {}
    value = data.get("base_model_name")
    return str(value) if value else "unknown"


def _column_names(connection: sa.Connection, table: str) -> set[str]:
    rows = connection.execute(sa.text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}


def _table_exists(connection: sa.Connection, table: str) -> bool:
    row = connection.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
        {"name": table},
    ).fetchone()
    return row is not None


def _load_existing_rows(connection: sa.Connection, root: Path) -> list[dict[str, object]]:
    if not _table_exists(connection, "trained_loras"):
        return []
    columns = _column_names(connection, "trained_loras")
    rows = connection.execute(sa.text("SELECT * FROM trained_loras ORDER BY id")).mappings().all()
    loaded: list[dict[str, object]] = []
    for row in rows:
        if "relative_path" in columns and row.get("relative_path") and row.get("weights_relpath"):
            relative_path = str(row["relative_path"])
            weights_relpath = str(row["weights_relpath"])
        elif "work_dir" in columns:
            work_dir = str(row["work_dir"])
            weights_path = str(row.get("weights_path") or work_dir)
            relative_path = normalize_lora_relative_path(work_dir, root)
            weights_relpath = to_lora_relative(root, weights_path) or normalize_lora_relative_path(
                weights_path, root
            )
            if not weights_relpath and relative_path:
                picked = pick_weights_file((root / relative_path).resolve())
                if picked is not None:
                    weights_relpath = to_lora_relative(root, picked)
        else:
            continue
        loaded.append(
            {
                "name": row["name"],
                "relative_path": relative_path,
                "weights_relpath": weights_relpath or relative_path,
                "job_id": row.get("job_id"),
                "config_id": row.get("config_id"),
                "config_yaml": row.get("config_yaml"),
                "base_model_name": row.get("base_model_name") or _base_model_from_yaml(row.get("config_yaml")),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
        )
    return loaded


def _discovered_rows(root: Path, now: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in LoraDiscoveryService(root=root).discover_lora_work_dirs():
        rows.append(
            {
                "name": item.name,
                "relative_path": item.relative_path,
                "weights_relpath": item.weights_relpath,
                "job_id": None,
                "config_id": None,
                "config_yaml": None,
                "base_model_name": "unknown",
                "created_at": now,
                "updated_at": now,
            }
        )
    return rows


def _merge_rows(
    existing: list[dict[str, object]],
    discovered: list[dict[str, object]],
    jobs: list[tuple[int, int | None, str | None, str | None]],
    root: Path,
) -> list[dict[str, object]]:
    by_path: dict[str, dict[str, object]] = {}
    for row in existing:
        canonical = str(row["relative_path"]).strip().strip("/\\").replace("\\", "/")
        by_path[canonical] = dict(row)

    for row in discovered:
        canonical = str(row["relative_path"]).strip().strip("/\\").replace("\\", "/")
        if canonical not in by_path:
            by_path[canonical] = dict(row)

    used_names = {str(row["name"]) for row in by_path.values()}
    for canonical, row in list(by_path.items()):
        if row.get("job_id") is not None:
            continue
        work_dir = (root / canonical).resolve()
        for job_id, config_id, config_yaml, output_path in jobs:
            if not output_path:
                continue
            try:
                resolved_output = Path(str(output_path)).expanduser().resolve()
            except OSError:
                continue
            if resolved_output != work_dir:
                continue
            row["job_id"] = job_id
            row["config_id"] = config_id
            row["config_yaml"] = config_yaml
            row["base_model_name"] = _base_model_from_yaml(config_yaml)
            break

        name = str(row["name"])
        if name in used_names and sum(1 for other in by_path.values() if other["name"] == name) > 1:
            candidate = name
            suffix = 2
            while candidate in used_names:
                candidate = f"{name}-{suffix}"
                suffix += 1
            row["name"] = candidate
            used_names.add(candidate)

    names_seen: set[str] = set()
    merged: list[dict[str, object]] = []
    for row in by_path.values():
        name = str(row["name"])
        candidate = name
        suffix = 2
        while candidate in names_seen:
            candidate = f"{name}-{suffix}"
            suffix += 1
        row["name"] = candidate
        names_seen.add(candidate)
        merged.append(row)
    return merged


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("DROP TABLE IF EXISTS _alembic_tmp_trained_loras"))
    connection.execute(sa.text("DROP TABLE IF EXISTS trained_loras_new"))

    root = lora_root_from_config()
    now = datetime.now(timezone.utc).isoformat()

    jobs = connection.execute(
        sa.text(
            "SELECT id, config_id, config_yaml, output_path FROM jobs "
            "WHERE job_type = 'training' AND status = 'completed' ORDER BY id"
        )
    ).fetchall()

    merged_rows = _merge_rows(
        _load_existing_rows(connection, root),
        _discovered_rows(root, now),
        jobs,
        root,
    )

    if _table_exists(connection, "trained_loras"):
        op.drop_table("trained_loras")

    op.create_table(
        "trained_loras",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("relative_path", sa.String(), nullable=False),
        sa.Column("weights_relpath", sa.String(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("config_id", sa.Integer(), nullable=True),
        sa.Column("config_yaml", sa.String(), nullable=True),
        sa.Column("base_model_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_trained_loras_name", "trained_loras", ["name"], unique=False)
    op.create_index("ix_trained_loras_base_model_name", "trained_loras", ["base_model_name"], unique=False)
    op.create_index("ix_trained_loras_config_id", "trained_loras", ["config_id"], unique=False)
    op.create_index("ix_trained_loras_job_id", "trained_loras", ["job_id"], unique=True)

    for row in merged_rows:
        connection.execute(
            sa.text(
                "INSERT INTO trained_loras "
                "(name, relative_path, weights_relpath, job_id, config_id, config_yaml, "
                "base_model_name, created_at, updated_at) "
                "VALUES (:name, :relative_path, :weights_relpath, :job_id, :config_id, :config_yaml, "
                ":base_model_name, :created_at, :updated_at)"
            ),
            {
                "name": row["name"],
                "relative_path": row["relative_path"],
                "weights_relpath": row["weights_relpath"],
                "job_id": row.get("job_id"),
                "config_id": row.get("config_id"),
                "config_yaml": row.get("config_yaml"),
                "base_model_name": row.get("base_model_name") or "unknown",
                "created_at": row.get("created_at") or now,
                "updated_at": row.get("updated_at") or now,
            },
        )


def downgrade() -> None:
    connection = op.get_bind()
    root = lora_root_from_config()
    now = datetime.now(timezone.utc).isoformat()

    rows = connection.execute(
        sa.text(
            "SELECT name, relative_path, weights_relpath, job_id, config_id, config_yaml, "
            "base_model_name, created_at, updated_at FROM trained_loras ORDER BY id"
        )
    ).mappings().all()

    op.create_table(
        "trained_loras_old",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("config_id", sa.Integer(), nullable=True),
        sa.Column("config_yaml", sa.String(), nullable=False),
        sa.Column("base_model_name", sa.String(), nullable=False),
        sa.Column("weights_path", sa.String(), nullable=False),
        sa.Column("work_dir", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["job_configs.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )

    for row in rows:
        work_dir = str((root / str(row["relative_path"])).resolve())
        weights_path = str((root / str(row["weights_relpath"])).resolve())
        connection.execute(
            sa.text(
                "INSERT INTO trained_loras_old "
                "(name, job_id, config_id, config_yaml, base_model_name, weights_path, work_dir, "
                "created_at, updated_at) "
                "VALUES (:name, :job_id, :config_id, :config_yaml, :base_model_name, :weights_path, "
                ":work_dir, :created_at, :updated_at)"
            ),
            {
                "name": row["name"],
                "job_id": row["job_id"] if row["job_id"] is not None else 0,
                "config_id": row["config_id"],
                "config_yaml": row["config_yaml"] or "",
                "base_model_name": row["base_model_name"],
                "weights_path": weights_path,
                "work_dir": work_dir,
                "created_at": row["created_at"] or now,
                "updated_at": row["updated_at"] or now,
            },
        )

    op.drop_table("trained_loras")
    op.rename_table("trained_loras_old", "trained_loras")
