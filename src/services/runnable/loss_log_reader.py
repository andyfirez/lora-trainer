"""Read training loss metrics from SQLite loss_log.db."""

import sqlite3
from pathlib import Path

from src.services.runnable.schemas import JobLossBatchResponse, JobLossResponse, LossPoint


def _fetch_metric_keys(con: sqlite3.Connection) -> list[str]:
    rows = con.execute("SELECT key FROM metric_keys ORDER BY key ASC").fetchall()
    return [row[0] for row in rows]


def _fetch_loss_points(
    con: sqlite3.Connection,
    *,
    key: str,
    limit: int,
    since_step: int | None,
    stride: int,
) -> list[LossPoint]:
    rows = con.execute(
        """
        SELECT
            m.step AS step,
            s.wall_time AS wall_time,
            m.value_real AS value,
            m.value_text AS value_text
        FROM metrics m
        JOIN steps s ON s.step = m.step
        WHERE m.key = ?
            AND (? IS NULL OR m.step > ?)
            AND (m.step % ?) = 0
        ORDER BY m.step ASC
        LIMIT ?
        """,
        (key, since_step, since_step, stride, limit),
    ).fetchall()
    return [
        LossPoint(
            step=row[0],
            wall_time=row[1],
            value=row[2] if row[2] is not None else (float(row[3]) if row[3] is not None else None),
        )
        for row in rows
    ]


def read_loss_log(
    log_path: Path,
    *,
    key: str = "loss/loss",
    limit: int = 2000,
    since_step: int | None = None,
    stride: int = 1,
) -> JobLossResponse:
    if not log_path.is_file():
        return JobLossResponse(key=key, keys=[], points=[])

    limit = min(limit, 20000)
    stride = max(1, stride)

    con = sqlite3.connect(str(log_path), timeout=30.0)
    con.execute("PRAGMA busy_timeout=30000;")
    try:
        keys = _fetch_metric_keys(con)
        points = _fetch_loss_points(con, key=key, limit=limit, since_step=since_step, stride=stride)
        return JobLossResponse(key=key, keys=keys, points=points)
    finally:
        con.close()


def read_loss_logs_batch(
    log_path: Path,
    *,
    key_queries: list[tuple[str, int | None]],
    limit: int = 2000,
    stride: int = 1,
) -> JobLossBatchResponse:
    if not log_path.is_file():
        return JobLossBatchResponse(keys=[], series={})

    limit = min(limit, 1_000_000)
    stride = max(1, stride)

    con = sqlite3.connect(str(log_path), timeout=30.0)
    con.execute("PRAGMA busy_timeout=30000;")
    try:
        keys = _fetch_metric_keys(con)
        series: dict[str, list[LossPoint]] = {}
        for key, since_step in key_queries:
            series[key] = _fetch_loss_points(
                con,
                key=key,
                limit=limit,
                since_step=since_step,
                stride=stride,
            )
        return JobLossBatchResponse(keys=keys, series=series)
    finally:
        con.close()
