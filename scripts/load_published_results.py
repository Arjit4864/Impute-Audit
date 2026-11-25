"""Index the published unit results (results/units) into runs/imputeaudit.sqlite for the Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imputeaudit import config  # noqa: E402
from imputeaudit.storage.repository import Repository  # noqa: E402

JSON_COLUMNS = ("class_tau", "class_top_k", "class_recall")


def main() -> None:
    units = sorted((ROOT / "results" / "units").glob("*.parquet"))
    versions = json.loads((ROOT / "results" / "environment.json").read_text())
    repo = Repository(config.RUNS_DIR)
    for path in units:
        unit_id = path.stem
        frame = pd.read_parquet(path)
        rows = frame.to_dict(orient="records")
        for row in rows:
            for col in JSON_COLUMNS:
                row[col] = json.loads(row[col])
        first = rows[0]
        cfg = {"dataset": first["dataset"], "mechanism": first["mechanism"], "rate": first["rate"],
               "seed": int(first["seed"])}
        repo.start_experiment(unit_id, first["dataset"], cfg, versions)
        repo.save_results(unit_id, rows)
        repo.save_artifact(unit_id, "unit_results", str(path.relative_to(ROOT)))
    print(f"{len(units)} units indexed in {repo.db_path}")


if __name__ == "__main__":
    main()
