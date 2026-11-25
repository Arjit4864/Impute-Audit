"""Task-appropriate prediction metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, n_classes: int) -> dict:
    pred = model.predict(X_test)
    metrics = {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "class_recall": recall_score(y_test, pred, average=None, labels=np.arange(n_classes)).tolist(),
    }
    if n_classes == 2:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["pr_auc"] = float(average_precision_score(y_test, proba))
        metrics["mcc"] = float(matthews_corrcoef(y_test, pred))
    return metrics
