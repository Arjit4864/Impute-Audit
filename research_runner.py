"""Offline research runner: executes the full experiment matrix with checkpointing.

Usage:
    python research_runner.py                 # full matrix, all CPU cores but two
    python research_runner.py --seeds 2 --datasets satimage   # quick pilot

Each experiment unit (dataset, mechanism, rate, seed) writes one Parquet checkpoint in
runs/units/. Units that already have a checkpoint are skipped, so the runner can be
stopped and restarted at any time. The main process alone writes to SQLite.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from imputeaudit import config
from imputeaudit.config import AppSettings
from imputeaudit.logging_setup import get_logger
from imputeaudit.storage.repository import Repository

MODELS_BY_DATASET = {
    "dry_bean": ("random_forest", "xgboost"),
    "pendigits": ("random_forest",),
    "satimage": ("random_forest",),
    "letter": ("random_forest",),
    "htru2": ("random_forest",),
}
COST_ORDER = ["letter", "dry_bean", "htru2", "pendigits", "satimage"]

_PREPARED: dict = {}


def package_versions() -> dict:
    import numpy, sklearn, shap, xgboost, scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "shap": shap.__version__, "xgboost": xgboost.__version__,
            "scipy": scipy.__version__}


def unit_path(units_dir: Path, unit_id: str) -> Path:
    return units_dir / f"{unit_id}.parquet"


def run_task(dataset: str, seed: int, units_dir: str) -> list[dict]:
    """Run every pending unit of one (dataset, seed); returns a status record per unit."""
    warnings.filterwarnings("ignore")
    import sklearn
    sklearn.set_config(working_memory=256)
    from imputeaudit.data.loader import load_bundled
    from imputeaudit.services.experiment_service import ExperimentConfig, ExperimentService, prepare

    units_dir = Path(units_dir)
    service = ExperimentService(AppSettings())
    if dataset not in _PREPARED:
        _PREPARED[dataset] = prepare(load_bundled(dataset), service.settings)
    prep = _PREPARED[dataset]
    prep.references.clear()
    models = MODELS_BY_DATASET[dataset]
    status = []

    for model in models:
        unit_id = f"{dataset}__{model}__clean_retrain__s{seed}"
        if unit_path(units_dir, unit_id).exists():
            continue
        start = time.perf_counter()
        try:
            row = service.clean_retrain_row(prep, model, seed)
            _write(units_dir, unit_id, [row])
            status.append({"unit_id": unit_id, "ok": True, "config": {"dataset": dataset, "model": model, "seed": seed,
                                                                      "condition": "clean_retrain"},
                           "seconds": time.perf_counter() - start})
        except Exception:
            status.append({"unit_id": unit_id, "ok": False, "error": traceback.format_exc(), "config": {"dataset": dataset}})

    for mechanism in config.MECHANISMS:
        for rate in config.RATES:
            cfg = ExperimentConfig(dataset, mechanism, rate, seed, config.IMPUTERS, models)
            unit_id = f"{dataset}__{mechanism}__r{int(rate * 100)}__s{seed}"
            if unit_path(units_dir, unit_id).exists():
                continue
            start = time.perf_counter()
            try:
                rows = service.run_unit(prep, cfg)
                _write(units_dir, unit_id, rows)
                status.append({"unit_id": unit_id, "ok": True, "config": cfg.to_dict(),
                               "seconds": time.perf_counter() - start})
            except Exception:
                status.append({"unit_id": unit_id, "ok": False, "error": traceback.format_exc(), "config": cfg.to_dict()})
    return status


def _write(units_dir: Path, unit_id: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    for col in ("class_tau", "class_top_k", "class_recall"):
        frame[col] = frame[col].apply(json.dumps)
    tmp = unit_path(units_dir, unit_id).with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, unit_path(units_dir, unit_id))


def ingest(repo: Repository, units_dir: Path, records: list[dict], versions: dict) -> None:
    for rec in records:
        dataset_id = rec["config"].get("dataset", "")
        repo.start_experiment(rec["unit_id"], dataset_id, rec["config"], versions)
        if rec["ok"]:
            frame = pd.read_parquet(unit_path(units_dir, rec["unit_id"]))
            rows = frame.to_dict(orient="records")
            for row in rows:
                for col in ("class_tau", "class_top_k", "class_recall"):
                    row[col] = json.loads(row[col])
            repo.save_results(rec["unit_id"], rows)
            repo.save_artifact(rec["unit_id"], "unit_results", str(unit_path(units_dir, rec["unit_id"])))
        else:
            repo.log_failure(rec["unit_id"], rec["error"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--datasets", nargs="*", default=COST_ORDER)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    log = get_logger("imputeaudit.runner")
    repo = Repository(config.RUNS_DIR)
    units_dir = config.RUNS_DIR / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    (config.RUNS_DIR / "environment.json").write_text(json.dumps(versions, indent=2))

    tasks = [(d, s) for d in args.datasets for s in range(args.seeds)]
    log.info("starting %d tasks with %d workers", len(tasks), args.jobs)
    print(f"{len(tasks)} tasks, {args.jobs} workers", flush=True)
    started = time.perf_counter()
    done = 0
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered", verbose=0)(
        delayed(run_task)(d, s, str(units_dir)) for d, s in tasks)
    for records in results:
        ingest(repo, units_dir, records, versions)
        done += 1
        failed = [r["unit_id"] for r in records if not r["ok"]]
        for unit in failed:
            log.error("unit failed: %s", unit)
        print(f"[{time.perf_counter() - started:7.0f}s] task {done}/{len(tasks)} done, "
              f"{len(records)} units, {len(failed)} failed", flush=True)
    print(f"{repo.count_completed()} experiment units completed and indexed", flush=True)


if __name__ == "__main__":
    main()
