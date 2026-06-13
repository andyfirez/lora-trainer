"""SQLite metric logger for UI loss graphs (ai-toolkit UILogger compatible schema)."""

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from src.trainer.config import TrainConfig


def build_loss_log_path(config: TrainConfig) -> Path:
    return Path(config.output_dir) / config.lora_name / "loss_log.db"


class MetricLogger:
    def __init__(
        self,
        log_file: Path | str,
        flush_every_n: int = 256,
        flush_every_secs: float = 0.25,
    ) -> None:
        self.log_file = Path(log_file)
        self._log_to_commit: dict[str, Any] = {}
        self._con: Optional[sqlite3.Connection] = None
        self._started = False
        self._step_counter = 0
        self._pending_steps: list[tuple[int, float]] = []
        self._pending_metrics: list[tuple[int, str, Optional[float], Optional[str]]] = []
        self._pending_key_minmax: dict[str, tuple[int, int]] = {}
        self._flush_every_n = int(flush_every_n)
        self._flush_every_secs = float(flush_every_secs)
        self._last_flush = time.time()
        self._first_commit_done = False

    def start(self) -> None:
        if self._started:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.log_file), timeout=30.0, isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL;")
        self._con.execute("PRAGMA synchronous=NORMAL;")
        self._con.execute("PRAGMA temp_store=MEMORY;")
        self._con.execute("PRAGMA foreign_keys=ON;")
        self._con.execute("PRAGMA busy_timeout=30000;")
        self._init_schema(self._con)
        self._started = True
        self._last_flush = time.time()

    def log(self, log_dict: dict[str, Any]) -> None:
        if not isinstance(log_dict, dict):
            raise TypeError("log_dict must be a dict")
        self._log_to_commit.update(log_dict)

    def commit(self, step: Optional[int] = None) -> None:
        if not self._started:
            self.start()
        if not self._log_to_commit:
            return
        if step is None:
            step = self._step_counter
            self._step_counter += 1
        else:
            step = int(step)
            if step >= self._step_counter:
                self._step_counter = step + 1
        if not self._first_commit_done:
            self._prune_future_steps(step)
            self._first_commit_done = True
        wall_time = time.time()
        self._pending_steps.append((step, wall_time))
        for k, v in self._log_to_commit.items():
            key = k if isinstance(k, str) else str(k)
            vr, vt = self._coerce_value(v)
            self._pending_metrics.append((step, key, vr, vt))
            if key in self._pending_key_minmax:
                lo, hi = self._pending_key_minmax[key]
                if step < lo:
                    lo = step
                if step > hi:
                    hi = step
                self._pending_key_minmax[key] = (lo, hi)
            else:
                self._pending_key_minmax[key] = (step, step)
        self._log_to_commit = {}
        now = time.time()
        if (
            len(self._pending_metrics) >= self._flush_every_n
            or (now - self._last_flush) >= self._flush_every_secs
        ):
            self._flush()

    def finish(self) -> None:
        if not self._started:
            return
        self._flush()
        assert self._con is not None
        self._con.close()
        self._con = None
        self._started = False

    def _init_schema(self, con: sqlite3.Connection) -> None:
        con.execute("BEGIN;")
        con.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step      INTEGER PRIMARY KEY,
                wall_time REAL NOT NULL
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS metric_keys (
                key             TEXT PRIMARY KEY,
                first_seen_step INTEGER,
                last_seen_step  INTEGER
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                step       INTEGER NOT NULL,
                key        TEXT NOT NULL,
                value_real REAL,
                value_text TEXT,
                PRIMARY KEY (step, key),
                FOREIGN KEY (step) REFERENCES steps(step) ON DELETE CASCADE
            );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_metrics_key_step ON metrics (key, step);")
        con.execute("COMMIT;")

    def _coerce_value(self, v: Any) -> tuple[Optional[float], Optional[str]]:
        if v is None:
            return None, None
        if isinstance(v, bool):
            return float(int(v)), None
        if isinstance(v, (int, float)):
            return float(v), None
        try:
            return float(v), None
        except (TypeError, ValueError):
            return None, str(v)

    def _prune_future_steps(self, current_step: int) -> None:
        assert self._con is not None
        con = self._con
        con.execute("BEGIN;")
        con.execute("DELETE FROM steps WHERE step > ?;", (current_step,))
        con.execute(
            "DELETE FROM metric_keys "
            "WHERE NOT EXISTS (SELECT 1 FROM metrics WHERE metrics.key = metric_keys.key);"
        )
        con.execute(
            "UPDATE metric_keys "
            "SET last_seen_step = (SELECT MAX(step) FROM metrics WHERE metrics.key = metric_keys.key) "
            "WHERE last_seen_step > ?;",
            (current_step,),
        )
        con.execute("COMMIT;")

    def _flush(self) -> None:
        if not self._pending_steps and not self._pending_metrics:
            return
        assert self._con is not None
        con = self._con
        con.execute("BEGIN;")
        if self._pending_steps:
            con.executemany(
                "INSERT INTO steps(step, wall_time) VALUES(?, ?) "
                "ON CONFLICT(step) DO UPDATE SET wall_time=excluded.wall_time;",
                self._pending_steps,
            )
        if self._pending_key_minmax:
            con.executemany(
                "INSERT INTO metric_keys(key, first_seen_step, last_seen_step) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "first_seen_step=MIN(metric_keys.first_seen_step, excluded.first_seen_step), "
                "last_seen_step=MAX(metric_keys.last_seen_step, excluded.last_seen_step);",
                [(k, lo, hi) for k, (lo, hi) in self._pending_key_minmax.items()],
            )
        if self._pending_metrics:
            con.executemany(
                "INSERT INTO metrics(step, key, value_real, value_text) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(step, key) DO UPDATE SET "
                "value_real=excluded.value_real, value_text=excluded.value_text;",
                self._pending_metrics,
            )
        con.execute("COMMIT;")
        self._pending_steps.clear()
        self._pending_metrics.clear()
        self._pending_key_minmax.clear()
        self._last_flush = time.time()
