"""Dataset profile: shape, types, class balance and feature summaries."""
from __future__ import annotations

import pandas as pd

from imputeaudit.data.loader import ValidatedDataset


def profile(dataset: ValidatedDataset) -> dict:
    counts = dataset.y.value_counts().sort_index()
    summary = dataset.X.describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
    summary["missing_share"] = dataset.X.isna().mean()
    return {
        "rows": len(dataset.X),
        "features": dataset.X.shape[1],
        "classes": dataset.n_classes,
        "class_counts": counts.to_dict(),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "missing_cells": int(dataset.X.isna().sum().sum()),
        "has_official_split": dataset.official_split is not None,
        "feature_summary": summary,
    }


def class_table(dataset: ValidatedDataset) -> pd.DataFrame:
    counts = dataset.y.value_counts().sort_index()
    return pd.DataFrame({"class": counts.index, "rows": counts.values,
                         "share": (counts.values / counts.sum()).round(4)})
