"""Default settings and supported experiment options for ImputeAudit.

Values can be overridden with environment variables prefixed IMPUTEAUDIT_ so the
same code runs locally, in Docker and in the offline research runner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data" / "samples"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"IMPUTEAUDIT_{name}", default)


RUNS_DIR = Path(_env("RUNS_DIR", str(APP_ROOT / "runs")))
LOG_DIR = Path(_env("LOG_DIR", str(APP_ROOT / "logs")))
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "50"))
MAX_ROWS = int(_env("MAX_ROWS", "200000"))
MIN_ROWS_PER_CLASS = int(_env("MIN_ROWS_PER_CLASS", "20"))
STORE_RAW_UPLOADS = _env("STORE_RAW_UPLOADS", "false").lower() == "true"

MECHANISMS = ("MCAR", "MAR", "MNAR")
RATES = (0.2, 0.4)
IMPUTERS = ("median", "knn", "iterative")
MODELS = ("random_forest", "xgboost")

TEST_SIZE = 0.2
LOGISTIC_SLOPE = 1.5
RATE_TOLERANCE = 0.02
KNN_NEIGHBORS = 5
ITERATIVE_MAX_ITER = 10
TOP_K = 5
EXPLAIN_ROWS_TOTAL = 300
MIN_EXPLAIN_ROWS_PER_CLASS = 10


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file: str
    target: str
    split: str
    title: str
    domain: str


DATASETS: dict[str, DatasetSpec] = {
    "dry_bean": DatasetSpec("dry_bean", "dry_bean.csv", "Class", "stratified", "Dry Bean", "agriculture / computer vision"),
    "pendigits": DatasetSpec("pendigits", "pendigits.csv", "digit", "official", "Pen-Based Digits", "handwriting (pen trajectories)"),
    "satimage": DatasetSpec("satimage", "satimage.csv", "land_cover", "official", "Statlog Landsat Satellite", "remote sensing"),
    "letter": DatasetSpec("letter", "letter.csv", "letter", "official", "Letter Recognition", "character images"),
    "htru2": DatasetSpec("htru2", "htru2.csv", "is_pulsar", "stratified", "HTRU2", "radio astronomy (binary)"),
}


@dataclass(frozen=True)
class ModelSettings:
    rf_trees: int = 100
    rf_max_depth: int | None = 12
    rf_min_samples_leaf: int = 2
    xgb_trees: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    n_jobs: int = 1


@dataclass(frozen=True)
class AppSettings:
    test_size: float = TEST_SIZE
    slope: float = LOGISTIC_SLOPE
    rate_tolerance: float = RATE_TOLERANCE
    knn_neighbors: int = KNN_NEIGHBORS
    iterative_max_iter: int = ITERATIVE_MAX_ITER
    top_k: int = TOP_K
    explain_rows_total: int = EXPLAIN_ROWS_TOTAL
    min_explain_rows_per_class: int = MIN_EXPLAIN_ROWS_PER_CLASS
    split_seed: int = 0
    models: ModelSettings = field(default_factory=ModelSettings)
