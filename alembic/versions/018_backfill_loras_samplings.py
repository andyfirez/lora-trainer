"""One-time DML: copy jobs/trained_loras data into loras/samplings.

Pure data migration — no `src.services.*` imports, no schema changes.
"""

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_JOB_TO_RUNNABLE_STATUS: dict[str, str] = {
    "pending": "draft",
    # A job that was queued/running when the app last stopped has no live
    # subprocess anymore — surface it as orphaned rather than silently "queued".
    "queued": "orphan",
    "running": "orphan",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}


def _map_status(job_status: str | None) -> str:
    return _JOB_TO_RUNNABLE_STATUS.get(job_status or "", "orphan")


def _dedupe_name(name: str, used_names: set[str]) -> str:
    base = name or "lora"
    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def upgrade() -> None:
    connection = op.get_bind()
    now = datetime.now(timezone.utc).isoformat()
    used_lora_names: set[str] = set()

    jobs_by_id: dict[int, sa.RowMapping] = {
        row["id"]: row
        for row in connection.execute(sa.text("SELECT * FROM jobs")).mappings().all()
    }

    # 1) trained_loras -> loras (always status=completed; enrich with job runtime fields).
    trained_loras = connection.execute(sa.text("SELECT * FROM trained_loras ORDER BY id")).mappings().all()
    migrated_job_ids: set[int] = set()
    for row in trained_loras:
        job = jobs_by_id.get(row["job_id"]) if row["job_id"] is not None else None
        if job is not None:
            migrated_job_ids.add(job["id"])
        name = _dedupe_name(str(row["name"]), used_lora_names)
        connection.execute(
            sa.text(
                """
                INSERT INTO loras (
                    name, status, config_yaml, queue_position, error_message, output_path, log_path, pid,
                    running_started_at, accumulated_elapsed_seconds,
                    relative_path, weights_relpath, base_model_name,
                    progress_step, progress_total, progress_loss, progress_avr_loss,
                    progress_epoch, progress_epoch_total, cache_progress_step, cache_progress_total,
                    last_checkpoint_path, last_checkpoint_epoch, last_checkpoint_step,
                    resume_checkpoint_path, resume_from_epoch, resume_from_step, save_checkpoint_requested,
                    created_at, updated_at
                ) VALUES (
                    :name, 'completed', :config_yaml, NULL, NULL, :output_path, :log_path, NULL,
                    NULL, :accumulated_elapsed_seconds,
                    :relative_path, :weights_relpath, :base_model_name,
                    :progress_step, :progress_total, :progress_loss, :progress_avr_loss,
                    :progress_epoch, :progress_epoch_total, :cache_progress_step, :cache_progress_total,
                    :last_checkpoint_path, :last_checkpoint_epoch, :last_checkpoint_step,
                    :resume_checkpoint_path, :resume_from_epoch, :resume_from_step, :save_checkpoint_requested,
                    :created_at, :updated_at
                )
                """
            ),
            {
                "name": name,
                "config_yaml": row["config_yaml"],
                "output_path": job["output_path"] if job else None,
                "log_path": job["log_path"] if job else None,
                "accumulated_elapsed_seconds": (job["accumulated_elapsed_seconds"] if job else 0.0) or 0.0,
                "relative_path": row["relative_path"],
                "weights_relpath": row["weights_relpath"],
                "base_model_name": row["base_model_name"] or "unknown",
                "progress_step": job["progress_step"] if job else None,
                "progress_total": job["progress_total"] if job else None,
                "progress_loss": job["progress_loss"] if job else None,
                "progress_avr_loss": job["progress_avr_loss"] if job else None,
                "progress_epoch": job["progress_epoch"] if job else None,
                "progress_epoch_total": job["progress_epoch_total"] if job else None,
                "cache_progress_step": job["cache_progress_step"] if job else None,
                "cache_progress_total": job["cache_progress_total"] if job else None,
                "last_checkpoint_path": job["last_checkpoint_path"] if job else None,
                "last_checkpoint_epoch": job["last_checkpoint_epoch"] if job else None,
                "last_checkpoint_step": job["last_checkpoint_step"] if job else None,
                "resume_checkpoint_path": job["resume_checkpoint_path"] if job else None,
                "resume_from_epoch": job["resume_from_epoch"] if job else None,
                "resume_from_step": job["resume_from_step"] if job else None,
                "save_checkpoint_requested": bool(job["save_checkpoint_requested"]) if job else False,
                "created_at": row["created_at"] or now,
                "updated_at": row["updated_at"] or now,
            },
        )

    # 2) training jobs with no trained_loras row (in progress / failed / cancelled / never linked).
    training_jobs = [
        row
        for row in jobs_by_id.values()
        if row["job_type"] == "training" and row["id"] not in migrated_job_ids
    ]
    for row in sorted(training_jobs, key=lambda r: r["id"]):
        name = _dedupe_name(str(row["name"]), used_lora_names)
        connection.execute(
            sa.text(
                """
                INSERT INTO loras (
                    name, status, config_yaml, queue_position, error_message, output_path, log_path, pid,
                    running_started_at, accumulated_elapsed_seconds,
                    relative_path, weights_relpath, base_model_name,
                    progress_step, progress_total, progress_loss, progress_avr_loss,
                    progress_epoch, progress_epoch_total, cache_progress_step, cache_progress_total,
                    last_checkpoint_path, last_checkpoint_epoch, last_checkpoint_step,
                    resume_checkpoint_path, resume_from_epoch, resume_from_step, save_checkpoint_requested,
                    created_at, updated_at
                ) VALUES (
                    :name, :status, :config_yaml, NULL, :error_message, :output_path, :log_path, NULL,
                    NULL, :accumulated_elapsed_seconds,
                    '', '', :base_model_name,
                    :progress_step, :progress_total, :progress_loss, :progress_avr_loss,
                    :progress_epoch, :progress_epoch_total, :cache_progress_step, :cache_progress_total,
                    :last_checkpoint_path, :last_checkpoint_epoch, :last_checkpoint_step,
                    :resume_checkpoint_path, :resume_from_epoch, :resume_from_step, :save_checkpoint_requested,
                    :created_at, :updated_at
                )
                """
            ),
            {
                "name": name,
                "status": _map_status(row["status"]),
                "config_yaml": row["config_yaml"],
                "error_message": row["error_message"],
                "output_path": row["output_path"],
                "log_path": row["log_path"],
                "accumulated_elapsed_seconds": row["accumulated_elapsed_seconds"] or 0.0,
                "base_model_name": "unknown",
                "progress_step": row["progress_step"],
                "progress_total": row["progress_total"],
                "progress_loss": row["progress_loss"],
                "progress_avr_loss": row["progress_avr_loss"],
                "progress_epoch": row["progress_epoch"],
                "progress_epoch_total": row["progress_epoch_total"],
                "cache_progress_step": row["cache_progress_step"],
                "cache_progress_total": row["cache_progress_total"],
                "last_checkpoint_path": row["last_checkpoint_path"],
                "last_checkpoint_epoch": row["last_checkpoint_epoch"],
                "last_checkpoint_step": row["last_checkpoint_step"],
                "resume_checkpoint_path": row["resume_checkpoint_path"],
                "resume_from_epoch": row["resume_from_epoch"],
                "resume_from_step": row["resume_from_step"],
                "save_checkpoint_requested": bool(row["save_checkpoint_requested"]),
                "created_at": row["created_at"] or now,
                "updated_at": row["updated_at"] or now,
            },
        )

    # 3) sampling jobs -> samplings.
    used_sampling_names: set[str] = set()
    sampling_jobs = [row for row in jobs_by_id.values() if row["job_type"] == "sampling"]
    for row in sorted(sampling_jobs, key=lambda r: r["id"]):
        name = _dedupe_name(str(row["name"]), used_sampling_names)
        connection.execute(
            sa.text(
                """
                INSERT INTO samplings (
                    name, status, config_yaml, queue_position, error_message, output_path, log_path, pid,
                    running_started_at, accumulated_elapsed_seconds,
                    lora_paths_yaml, progress_step, progress_total, progress_status,
                    created_at, updated_at
                ) VALUES (
                    :name, :status, :config_yaml, NULL, :error_message, :output_path, :log_path, NULL,
                    NULL, :accumulated_elapsed_seconds,
                    :lora_paths_yaml, :progress_step, :progress_total, :progress_status,
                    :created_at, :updated_at
                )
                """
            ),
            {
                "name": name,
                "status": _map_status(row["status"]),
                "config_yaml": row["config_yaml"],
                "error_message": row["error_message"],
                "output_path": row["output_path"],
                "log_path": row["log_path"],
                "accumulated_elapsed_seconds": row["accumulated_elapsed_seconds"] or 0.0,
                "lora_paths_yaml": row["lora_paths_yaml"],
                "progress_step": row["progress_step"],
                "progress_total": row["progress_total"],
                "progress_status": row["progress_status"],
                "created_at": row["created_at"] or now,
                "updated_at": row["updated_at"] or now,
            },
        )


def downgrade() -> None:
    # Best-effort: data merged from two legacy tables cannot be split back losslessly.
    connection = op.get_bind()
    connection.execute(sa.text("DELETE FROM samplings"))
    connection.execute(sa.text("DELETE FROM loras"))
