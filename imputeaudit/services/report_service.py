"""Tables, figures and export bundles built from stored results."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px

from imputeaudit import config as app_config
from imputeaudit.evaluation.statistics import bootstrap_ci


def self_runs_dir() -> Path:
    return app_config.RUNS_DIR

METRICS = ["macro_f1", "nrmse", "global_tau", "mean_class_tau", "min_class_tau", "delta_tau",
           "global_top_k", "mean_class_top_k"]


class ReportService:
    def __init__(self, repository):
        self.repository = repository

    def load_results(self) -> pd.DataFrame:
        return self.repository.results_frame()

    @staticmethod
    def summary_table(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results
        keys = ["dataset", "model", "mechanism", "rate", "imputer"]
        rows = []
        for key, group in results.groupby(keys):
            row = dict(zip(keys, key))
            row["masks"] = len(group)
            for m in METRICS:
                row[m] = group[m].mean()
            low, high = bootstrap_ci(group["delta_tau"].to_numpy())
            row["delta_tau_ci"] = f"[{low:.3f}, {high:.3f}]"
            rows.append(row)
        return pd.DataFrame(rows).round(4)

    @staticmethod
    def tau_comparison_figure(results: pd.DataFrame):
        long = results.melt(id_vars=["mechanism", "rate", "imputer"], value_vars=["global_tau", "mean_class_tau", "min_class_tau"],
                            var_name="ranking", value_name="kendall_tau_b")
        return px.box(long, x="imputer", y="kendall_tau_b", color="ranking", facet_col="mechanism",
                      facet_row="rate", points=False, title="Ranking agreement with the complete-data reference")

    @staticmethod
    def export_bundle(results: pd.DataFrame, config: dict | None = None) -> bytes:
        buffer = io.BytesIO()
        summary = ReportService.summary_table(results)
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("results.csv", results.to_csv(index=False))
            bundle.writestr("summary.csv", summary.to_csv(index=False))
            bundle.writestr("results.json", results.to_json(orient="records", indent=1))
            bundle.writestr("config.json", json.dumps(config or {}, indent=2))
            env_file = Path(self_runs_dir()) / "environment.json"
            if env_file.exists():
                bundle.writestr("environment.json", env_file.read_text())
            html = (f"<html><head><meta charset='utf-8'><title>ImputeAudit report</title></head><body>"
                    f"<h1>ImputeAudit report</h1><p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
                    f"{len(results)} result rows.</p>{summary.to_html(index=False)}</body></html>")
            bundle.writestr("report.html", html)
        return buffer.getvalue()
