"""SQLite run index plus run-artefact folders.

Only the main process writes to SQLite; worker processes return plain dictionaries.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY, name TEXT, file_hash TEXT UNIQUE, row_count INTEGER,
    feature_count INTEGER, target_name TEXT, class_counts TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS experiments (
    config_id TEXT PRIMARY KEY, dataset_id TEXT REFERENCES datasets(dataset_id), config_json TEXT,
    status TEXT, error_message TEXT, package_versions TEXT, started_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT, config_id TEXT REFERENCES experiments(config_id),
    dataset TEXT, model TEXT, mechanism TEXT, rate REAL, seed INTEGER, imputer TEXT,
    achieved_rate_train REAL, achieved_rate_test REAL, nrmse REAL, macro_f1 REAL, balanced_accuracy REAL,
    ref_macro_f1 REAL, pr_auc REAL, mcc REAL, global_tau REAL, mean_class_tau REAL, min_class_tau REAL,
    delta_tau REAL, global_top_k REAL, mean_class_top_k REAL, min_class_top_k REAL,
    model_only_global_tau REAL, model_only_mean_class_tau REAL, class_tau TEXT, class_top_k TEXT,
    class_recall TEXT, runtime_s REAL);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT, config_id TEXT REFERENCES experiments(config_id),
    artifact_type TEXT, path TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_results_condition ON results(dataset, model, mechanism, rate, imputer);
"""

RESULT_COLUMNS = [
    "dataset", "model", "mechanism", "rate", "seed", "imputer", "achieved_rate_train", "achieved_rate_test",
    "nrmse", "macro_f1", "balanced_accuracy", "ref_macro_f1", "pr_auc", "mcc", "global_tau", "mean_class_tau",
    "min_class_tau", "delta_tau", "global_top_k", "mean_class_top_k", "min_class_top_k",
    "model_only_global_tau", "model_only_mean_class_tau", "class_tau", "class_top_k", "class_recall", "runtime_s",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runs_dir / "imputeaudit.sqlite"
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def save_dataset(self, dataset_id: str, name: str, file_hash: str, rows: int, features: int,
                     target: str, class_counts: dict) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO datasets VALUES (?,?,?,?,?,?,?,?)",
                         (dataset_id, name, file_hash, rows, features, target, json.dumps(class_counts), now()))

    def exists(self, config_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM experiments WHERE config_id=?", (config_id,)).fetchone()
        return bool(row and row[0] == "completed")

    def start_experiment(self, config_id: str, dataset_id: str, config: dict, versions: dict) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM results WHERE config_id=?", (config_id,))
            conn.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                         (config_id, dataset_id, json.dumps(config), "running", None, json.dumps(versions), now(), None))

    def save_results(self, config_id: str, rows: list[dict]) -> None:
        with self.connect() as conn:
            for row in rows:
                values = [json.dumps(row.get(c)) if isinstance(row.get(c), (list, dict)) else row.get(c)
                          for c in RESULT_COLUMNS]
                conn.execute(f"INSERT INTO results (config_id, {', '.join(RESULT_COLUMNS)}) "
                             f"VALUES (?{', ?' * len(RESULT_COLUMNS)})", [config_id, *values])
            conn.execute("UPDATE experiments SET status='completed', finished_at=? WHERE config_id=?", (now(), config_id))

    def log_failure(self, config_id: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE experiments SET status='failed', error_message=?, finished_at=? WHERE config_id=?",
                         (message[:2000], now(), config_id))

    def save_artifact(self, config_id: str, artifact_type: str, path: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO artifacts (config_id, artifact_type, path, created_at) VALUES (?,?,?,?)",
                         (config_id, artifact_type, path, now()))

    def results_frame(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM results", conn)

    def experiments_frame(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM experiments ORDER BY started_at DESC", conn)

    def count_completed(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiments WHERE status='completed'").fetchone()[0]
