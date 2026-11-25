"""Summary statistics across independent mask replicates."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_wilcoxon(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank on paired replicates, with rank-biserial effect size."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.abs(d) > 1e-12]
    if len(d) < 5:
        return {"n": int(len(d)), "statistic": float("nan"), "p": float("nan"), "rank_biserial": float("nan")}
    res = wilcoxon(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return {"n": int(len(d)), "statistic": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial": float((r_plus - r_minus) / (r_plus + r_minus))}


def benjamini_hochberg(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out
