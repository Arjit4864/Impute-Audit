"""Leakage-safe preprocessing: scaler and imputer fitted on training rows only."""
from __future__ import annotations

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_imputer(name: str, seed: int, knn_neighbors: int = 5, max_iter: int = 10):
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "knn":
        return KNNImputer(n_neighbors=knn_neighbors)
    if name == "iterative":
        return IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter, initial_strategy="median",
                                random_state=seed, tol=1e-3)
    raise ValueError(f"Unknown imputer: {name}")


def fit_preprocessor(X_train_masked: np.ndarray, imputer: str, seed: int, knn_neighbors: int = 5,
                     max_iter: int = 10) -> Pipeline:
    """Fit StandardScaler (NaN-aware) and the imputer on masked training rows only."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("impute", make_imputer(imputer, seed, knn_neighbors, max_iter)),
    ])
    return pipe.fit(X_train_masked)


def transform_test(pipe: Pipeline, X_test_masked: np.ndarray) -> np.ndarray:
    """Apply already fitted transforms; never calls fit on held-out rows."""
    return pipe.transform(X_test_masked)
