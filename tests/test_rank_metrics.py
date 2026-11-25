import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imputeaudit.explainability.rankings import rank_features, rank_order
from imputeaudit.explainability.shap_engine import compute_shap
from imputeaudit.explainability.stability import compare_rankings, kendall_tau_b, top_k_jaccard


def test_kendall_identical_reversed_tied():
    a = np.array([5.0, 4, 3, 2, 1])
    assert kendall_tau_b(a, a) == pytest.approx(1.0)
    assert kendall_tau_b(a, a[::-1]) == pytest.approx(-1.0)
    tied = np.array([5.0, 4, 4, 2, 1])
    assert 0.9 < kendall_tau_b(a, tied) < 1.0


def test_top5_jaccard_three_of_five():
    ref = np.array([10, 9, 8, 7, 6, 1, 1, 1], dtype=float)
    cand = np.array([10, 9, 8, 0, 0, 7, 6, 0], dtype=float)
    assert top_k_jaccard(ref, cand, 5) == 3 / 7


def test_shap_shape_rf_and_xgb():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    for model in (RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y),
                  XGBClassifier(n_estimators=10, max_depth=3).fit(X, y)):
        assert compute_shap(model, X[:20], 3).shape == (20, 5, 3)
    binary = XGBClassifier(n_estimators=10, max_depth=3).fit(X, (y > 0).astype(int))
    assert compute_shap(binary, X[:20], 2).shape == (20, 5, 2)


def test_global_is_equal_weight_mean():
    phi = np.zeros((4, 3, 2))
    phi[:, 0, 0] = 2.0
    phi[:, 1, 1] = -4.0
    r = rank_features(phi)
    assert np.allclose(r["global"], [1.0, 2.0, 0.0])
    assert rank_order(r["global"]).tolist() == [2, 1, 3]


def test_compare_with_itself_is_perfect():
    rng = np.random.default_rng(0)
    r = rank_features(rng.normal(size=(50, 8, 3)))
    out = compare_rankings(r, r)
    assert out["global_tau"] == pytest.approx(1.0) and out["mean_class_tau"] == pytest.approx(1.0)
    assert out["global_top_k"] == 1.0
    assert out["delta_tau"] == pytest.approx(0.0, abs=1e-12)

