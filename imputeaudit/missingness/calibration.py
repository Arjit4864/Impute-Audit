"""Intercept calibration so that logistic masks hit a requested missingness rate."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def calibrate_intercept(z: np.ndarray, slope: float, rate: float, iterations: int = 80) -> float:
    """Return a such that mean(sigmoid(a + slope * z)) equals ``rate`` (bisection)."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly between 0 and 1")
    lo, hi = -40.0, 40.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if expit(mid + slope * z).mean() < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
