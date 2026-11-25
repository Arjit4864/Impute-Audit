"""MCAR, MAR and MNAR mask generators.

All masks are boolean arrays of the predictor matrix only; the target is never an input.
Probabilities are computed from the original complete values before anything is hidden.
Standardization uses reference statistics (the training rows), so train and test masks
follow the same generating rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from imputeaudit.missingness.calibration import calibrate_intercept


@dataclass
class MaskResult:
    mask: np.ndarray
    mechanism: str
    rate: float
    achieved_rate: float
    seed: int
    anchor: int | None
    slope: float
    direction: str

    def apply(self, X: np.ndarray) -> np.ndarray:
        out = np.array(X, dtype=float, copy=True)
        out[self.mask] = np.nan
        return out

    def metadata(self) -> dict:
        return {"mechanism": self.mechanism, "rate": self.rate, "achieved_rate": round(self.achieved_rate, 5),
                "seed": self.seed, "anchor": self.anchor, "slope": self.slope, "direction": self.direction}


def choose_anchor(X_train: np.ndarray) -> int:
    """Pick the predictor with the highest mean absolute correlation with the others.

    A well-connected anchor makes the MAR mask informative about the hidden values,
    which is the situation in which MAR differs most from MCAR.
    """
    corr = np.abs(np.corrcoef(X_train, rowvar=False))
    np.fill_diagonal(corr, np.nan)
    return int(np.nanargmax(np.nanmean(corr, axis=0)))


def make_missingness_mask(X: np.ndarray, mechanism: str, rate: float, seed: int,
                          anchor: int | None = None, slope: float = 1.5,
                          ref_mean: np.ndarray | None = None, ref_std: np.ndarray | None = None) -> MaskResult:
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError("Masks are generated on complete data only.")
    rng = np.random.default_rng(seed)
    mean = X.mean(axis=0) if ref_mean is None else ref_mean
    std = X.std(axis=0) if ref_std is None else ref_std
    Z = (X - mean) / np.where(std > 0, std, 1.0)
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)
    mechanism = mechanism.upper()
    direction = "none"

    if mechanism == "MCAR":
        mask = rng.random((n, d)) < rate
        eligible = mask.size
    elif mechanism == "MAR":
        if anchor is None:
            raise ValueError("MAR needs an always-observed anchor predictor.")
        z = Z[:, anchor]
        p = expit(calibrate_intercept(z, slope, rate) + slope * z)
        for j in range(d):
            if j != anchor:
                mask[:, j] = rng.random(n) < p
        eligible = n * (d - 1)
        direction = "high anchor values" if slope > 0 else "low anchor values"
    elif mechanism == "MNAR":
        for j in range(d):
            z = Z[:, j]
            p = expit(calibrate_intercept(z, slope, rate) + slope * z)
            mask[:, j] = rng.random(n) < p
        eligible = mask.size
        direction = "high values" if slope > 0 else "low values"
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return MaskResult(mask=mask, mechanism=mechanism, rate=rate, achieved_rate=float(mask.sum() / eligible),
                      seed=seed, anchor=anchor if mechanism == "MAR" else None, slope=slope, direction=direction)
