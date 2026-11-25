"""Ranking agreement between a reference and a candidate explanation."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    value = kendalltau(a, b, variant="b").statistic
    return float(value) if np.isfinite(value) else 0.0


def top_k_jaccard(ref: np.ndarray, cand: np.ndarray, k: int = 5) -> float:
    k = min(k, len(ref))
    a = set(np.argsort(-ref, kind="stable")[:k])
    b = set(np.argsort(-cand, kind="stable")[:k])
    return len(a & b) / len(a | b)


def compare_rankings(reference: dict, candidate: dict, k: int = 5) -> dict:
    n_classes = reference["per_class"].shape[1]
    class_tau = [kendall_tau_b(reference["per_class"][:, c], candidate["per_class"][:, c]) for c in range(n_classes)]
    class_jac = [top_k_jaccard(reference["per_class"][:, c], candidate["per_class"][:, c], k) for c in range(n_classes)]
    global_tau = kendall_tau_b(reference["global"], candidate["global"])
    mean_class_tau = float(np.mean(class_tau))
    return {
        "global_tau": global_tau,
        "mean_class_tau": mean_class_tau,
        "min_class_tau": float(np.min(class_tau)),
        "delta_tau": global_tau - mean_class_tau,
        "global_top_k": top_k_jaccard(reference["global"], candidate["global"], k),
        "mean_class_top_k": float(np.mean(class_jac)),
        "min_class_top_k": float(np.min(class_jac)),
        "class_tau": class_tau,
        "class_top_k": class_jac,
    }
