"""Single adapter for TreeSHAP so every comparison uses identical settings."""
from __future__ import annotations

import numpy as np
import shap


def compute_shap(model, X_eval: np.ndarray, n_classes: int) -> np.ndarray:
    """Return SHAP values with shape (rows, features, classes).

    Uses TreeExplainer with the tree-path-dependent algorithm (no background sample),
    so the clean and the imputed model are explained in the same way. Random Forest
    attributions are on the probability scale; XGBoost attributions are on the margin
    (log-odds) scale. Comparisons are therefore made only within one model family.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_eval, check_additivity=False)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:
        values = np.stack([-values, values], axis=-1) if n_classes == 2 else values[..., np.newaxis]
    if values.shape[-1] != n_classes:
        raise ValueError(f"SHAP returned {values.shape[-1]} outputs for {n_classes} classes")
    return values
