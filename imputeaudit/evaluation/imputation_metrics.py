"""Reconstruction error of artificially hidden cells."""
from __future__ import annotations

import numpy as np


def nrmse(true_scaled: np.ndarray, imputed_scaled: np.ndarray, mask: np.ndarray) -> float:
    """RMSE over masked cells in training-standardized units (i.e. RMSE / training std)."""
    if not mask.any():
        return 0.0
    diff = (true_scaled - imputed_scaled)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))
