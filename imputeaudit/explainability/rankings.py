"""Global and class-wise feature importance from SHAP values."""
from __future__ import annotations

import numpy as np


def rank_features(phi: np.ndarray) -> dict:
    """phi has shape (rows, features, classes).

    I[j, c] = mean_i |phi[i, j, c]|            class-wise importance
    I_global[j] = mean_c I[j, c]               equal class weights (macro)
    """
    class_importance = np.abs(phi).mean(axis=0)
    return {"global": class_importance.mean(axis=1), "per_class": class_importance}


def rank_order(importance: np.ndarray) -> np.ndarray:
    """Rank 1 = most important."""
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(importance) + 1)
    return ranks
