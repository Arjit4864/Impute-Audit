import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from imputeaudit.config import AppSettings, ModelSettings  # noqa: E402
from imputeaudit.data.loader import validate_dataset  # noqa: E402


@pytest.fixture
def toy_frame():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 6))
    X[:, 1] = X[:, 0] * 0.8 + rng.normal(scale=0.3, size=n)
    y = np.where(X[:, 0] + X[:, 2] > 0.5, "a", np.where(X[:, 3] > 0, "b", "c"))
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    frame["label"] = y
    return frame


@pytest.fixture
def toy_dataset(toy_frame):
    ds = validate_dataset(toy_frame, "label", "toy", "hash")
    ds.meta["split"] = "stratified"
    return ds


@pytest.fixture
def small_settings():
    return AppSettings(explain_rows_total=60, min_explain_rows_per_class=10,
                       models=ModelSettings(rf_trees=20, rf_max_depth=6, xgb_trees=20, xgb_max_depth=3))
