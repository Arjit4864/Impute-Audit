"""Classifier construction and training."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imputeaudit.config import ModelSettings


def build_model(name: str, seed: int, n_classes: int, settings: ModelSettings = ModelSettings()):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=settings.rf_trees, max_depth=settings.rf_max_depth,
                                      min_samples_leaf=settings.rf_min_samples_leaf, random_state=seed,
                                      n_jobs=settings.n_jobs)
    if name == "xgboost":
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        return XGBClassifier(n_estimators=settings.xgb_trees, max_depth=settings.xgb_max_depth,
                             learning_rate=settings.xgb_learning_rate, subsample=0.8, colsample_bytree=0.8,
                             objective=objective, tree_method="hist", random_state=seed,
                             n_jobs=settings.n_jobs, verbosity=0)
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, X_train: np.ndarray, y_train: np.ndarray, seed: int, n_classes: int,
              settings: ModelSettings = ModelSettings()):
    model = build_model(name, seed, n_classes, settings)
    return model.fit(X_train, y_train)
